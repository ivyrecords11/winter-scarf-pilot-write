#@title Cerebellar Modification
#@title === 1) Imports & Utils ===
import math, textwrap, numpy as np
import mujoco as mj
from gymnasium import Env, spaces
from dataclasses import dataclass

# (Optional) SpikingJelly: 간단 레이트→포아송 인코더 예시용
try:
    from spikingjelly.activation_based import functional as sjF
except Exception:
    sjF = None

# ---- 4) 소뇌형 SNN (SpikingJelly 캡슐화) -------------------------------------
from typing import Optional, Tuple, Protocol
import torch
import torch.nn as nn
from math import ceil
from spikingjelly.activation_based import layer, neuron, functional
from spikingjelly.activation_based import learning, surrogate
DT = 1e-4   # 0.1 ms


#@title @dataclass SimulationConfig

@dataclass
class SimulationConfig:
    # --- 기본 시뮬레이션 ---
    dt_ms: float = 0.1                              # [ms]
    dt_sim : float = dt_ms / 1000
    plane_size: float = 0.30                        # [m] 한 변 길이 (±0.15 m)
    goal_r: float = 0.005                           # [m] 목표 반경
    tau_pre_ms: float = 5.0
    tau_post_ms: float = 5.0
    tau_e_ms: float = 50.0
    eta: float = 1e-3
    max_dw_per_step: float = 1e-3
    scale_every: int = 100
    use_mf_collat: bool = True
    mf_collat_gain: float = 0.3
    seed: int = 42

    # --- SNN / 센서 ---
    n_mf: int = 100                                 # 센서(모스피버, etc.) 개수
    sensor_slots_xy: int = 10                       # 10x10 = 100 슬롯
    conduction_vel: float = 50.0                    # [m/s] 신경 신호 속도
    use_manhattan_delay: bool = True                # 맨해튼 거리를 쓸 건가요?

    # --- 뉴런 계층 ---
    n_mf: int = 100
    n_grc: int = 1024
    n_pkj: int = 64
    n_motor: int = 4                                # +X, -X, +Y, -Y

    # --- 물리/액추에이터(고정 파라미터) ---
    plate_mass: float = 0.50                        # [kg] 예: 0.5 kg
    actuator_torque_scale: float = 0.05             # [N·m] 스칼라 가중치
    friction_coeff: float = 0.05                    # 무차원/단순 감쇠항
    gravity: float = 9.8
    # 난수 고정(슬롯 샘플링 재현성)
    seed: int = 42

    #inputsimconfig

    # 시뮬레이션
    # 발화(rate-based Poisson)
    base_rate_hz: float = 400.0
    rate_scale_weight: float = 40.0      # 가중: lam = base + k*weight_g
    sigma_ratio: float = 1.0             # σ = ratio * ball_radius
    # 판 기울기 제한
    max_tilt_deg: float = 5.0
    # 비디오/시각화
    video_fps: int = 30

    # --- 4-엣지 액추에이터 매핑(가상) ---
    edge_ctrl_min: float = 0.0   # 각 채널 입력 하한(수축 0~1 가정)
    edge_ctrl_max: float = 1.0   # 각 채널 입력 상한
    edge_gain_deg: float = 10.0   # (우-좌) 또는 (상-하) 1.0 차이가 만드는 기울기[deg]
    edge_x_sign: float = +1.0    # 부호 교정(축 정의에 따라 필요시 -1)
    edge_y_sign: float = +1.0

    # 안정화
    settle_steps: int = 500

class ISNN(Protocol):
    def reset(self) -> None: ...
    def forward(self, mf_spikes: np.ndarray) -> np.ndarray: ... # -> motor spikes (4,)
    def learn(self, cf_signal: float) -> None: ...

class CerebellarNet(ISNN):
    """
    내부 구현은 SpikingJelly activation_based(IFNode, Linear) 사용.  :contentReference[oaicite:8]{index=8}
    forward:  MF -> GrC -> PkG -> Motor(4) 스파이크
    learn:    PF(GrC)-Motor 가중치에 3-요소 규칙(eligibility × CF) 적용
    """
    def __init__(self, cfg: SimulationConfig, device: Optional[str] = None):
        self.cfg = cfg

        self.torch = torch; self.nn = nn;
        self.sj_func = functional       #spikingjelly functional
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---------- Fixed expansion: MF -> GrC ----------
        self.mf2grc = layer.Linear(cfg.n_mf, cfg.n_grc, bias=False).to(self.dev)
        with self.torch.no_grad():
            W = self.torch.zeros(cfg.n_grc, cfg.n_mf, device=self.dev)
            mask = (self.torch.rand_like(W) < 0.1).float()
            W += mask
            W /= (W.sum(dim=1, keepdim=True) + 1e-6)
            self.mf2grc.weight.copy_(W)
        for p in self.mf2grc.parameters(): p.requires_grad_(False)
        self.grc = neuron.IFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True).to(self.dev)

        # ---------- Trainable plastic site: GrC(PF) -> PKJ ----------
        self.grc2pkj = layer.Linear(cfg.n_grc, cfg.n_pkj, bias=False).to(self.dev)
        self.nn.init.zeros_(self.grc2pkj.weight)  # start neutral; learn via CF-gated 3-factor
        self.pkj = neuron.IFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True).to(self.dev)

        tau_pre_steps  = float(self.cfg.tau_pre_ms / self.cfg.dt_ms)
        tau_post_steps = float(self.cfg.tau_post_ms / self.cfg.dt_ms)
        def _f_w(x: torch.Tensor) -> torch.Tensor:
            return torch.clamp(x, -1.5, 1.5)
        self.stdp_pkj = learning.STDPLearner(
            step_mode='s',
            synapse=self.grc2pkj,    # 학습할 Linear
            sn=self.pkj,             # post 뉴런층
            tau_pre=tau_pre_steps,
            tau_post=tau_post_steps,
            f_pre=_f_w,
            f_post=_f_w
        )
        self.opt_stdp = torch.optim.SGD(self.grc2pkj.parameters(), lr=self.cfg.eta, momentum=0.0)

        # ---------- CF input: strong drive to PKJ (one-to-one by default) ----------
        self.cf2pkj = layer.Linear(cfg.n_pkj, cfg.n_pkj, bias=False).to(self.dev)
        with self.torch.no_grad():
            self.cf2pkj.weight.copy_(self.torch.eye(cfg.n_pkj, device=self.dev) * 3.0)  # strong drive
        for p in self.cf2pkj.parameters(): p.requires_grad_(False)

        # ---------- DN/motor: receives (-) PKJ, (+) MF collaterals (optional) ----------
        self.pkj2mo = layer.Linear(cfg.n_pkj, cfg.n_motor, bias=False).to(self.dev)
        with self.torch.no_grad():
            # inhibitory mapping: negative weights, broadly tuned
            Wp = - self.torch.randn(cfg.n_motor, cfg.n_pkj, device=self.dev)
            Wp /= (Wp.std(dim=1, keepdim=True) + 1e-6)
            self.pkj2mo.weight.copy_(Wp * 0.2)
        for p in self.pkj2mo.parameters(): p.requires_grad_(False)

        if cfg.use_mf_collat:
            self.mf2mo = layer.Linear(cfg.n_mf, cfg.n_motor, bias=False).to(self.dev)
            with self.torch.no_grad():
                Wc = self.torch.zeros(cfg.n_motor, cfg.n_mf, device=self.dev)
                mask = (self.torch.rand_like(Wc) < 0.2).float()
                Wc += mask
                Wc /= (Wc.sum(dim=1, keepdim=True) + 1e-6)
                self.mf2mo.weight.copy_(Wc * cfg.mf_collat_gain)
            for p in self.mf2mo.parameters(): p.requires_grad_(False)
        else:
            self.mf2mo = None

        self.mo  = neuron.IFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True).to(self.dev)

        # ---------- Eligibility traces for PF->PKJ ----------
        self.pre_tr  = np.zeros(cfg.n_grc, dtype=np.float32)     # PF (pre)
        self.post_tr = np.zeros(cfg.n_pkj, dtype=np.float32)     # PKJ (post)
        self.E = np.zeros((cfg.n_grc, cfg.n_pkj), dtype=np.float32)

        self.pre_decay  = math.exp(-cfg.dt_ms/cfg.tau_pre_ms)
        self.post_decay = math.exp(-cfg.dt_ms/cfg.tau_post_ms)
        self.e_decay    = math.exp(-cfg.dt_ms/cfg.tau_e_ms)

        # cache last spikes for learning step
        self._last = {}

    def reset(self) -> None:
        self.sj_func.reset_net(self.mf2grc); self.sj_func.reset_net(self.grc)
        self.sj_func.reset_net(self.grc2pkj); self.sj_func.reset_net(self.pkj)
        self.sj_func.reset_net(self.pkj2mo);  self.sj_func.reset_net(self.mo)
        if self.mf2mo is not None: self.sj_func.reset_net(self.mf2mo)
        self.pre_tr[:] = 0; self.post_tr[:] = 0; self.E[:] = 0
        self._last.clear()

    def forward(self, mf_spikes: np.ndarray, cf_spikes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        mf_spikes : (N_MF,)  binary
        cf_spikes : (N_PKJ,) binary or rate-like; if None, zeros
        returns   : (N_MOTOR,) binary spikes
        """
        t = self.torch
        x_mf = t.from_numpy(mf_spikes)[None,:].to(self.dev)
        x_cf = None
        if cf_spikes is not None:
            x_cf = t.from_numpy(cf_spikes)[None,:].to(self.dev)

        grc_spk = self.grc(self.mf2grc(x_mf))
        pkj_in  = self.grc2pkj(grc_spk)
        if x_cf is not None:
            pkj_in = pkj_in + self.cf2pkj(x_cf)   # CF drives PKJ strongly
        pkj_spk = self.pkj(pkj_in)

        mo_in = self.pkj2mo(pkj_spk)              # inhibition by PKJ
        if (self.mf2mo is not None):
            mo_in = mo_in + self.mf2mo(x_mf)      # MF collateral excitation
        mo_spk = self.mo(mo_in)

        # cache last spikes for learning
        self._last['grc'] = grc_spk.detach().cpu().numpy()[0]
        self._last['pkj'] = pkj_spk.detach().cpu().numpy()[0]
        self._last['mo']  = mo_spk.detach().cpu().numpy()[0]
        self._last['cf']  = (cf_spikes.copy() if cf_spikes is not None else None)
        return self._last['mo']

    def learn(self, cf_signal: Optional[np.ndarray] = None) -> None:
        """
        CF-gated three-factor at PF->PKJ:
          ΔW_ij ∝ η * (CF_j) * E_ij
        cf_signal:
          - None or scalar → uses scalar for all PKJ units
          - (N_PKJ,) vector → per-PKJ gating (recommended if you pass cf_spikes)
        """
        grc_out = self._last['grc']; pkj_out = self._last['pkj']
        # 1) update traces
        self.pre_tr  = self.pre_tr  * self.pre_decay  + grc_out
        self.post_tr = self.post_tr * self.post_decay + pkj_out
        self.E = self.E * self.e_decay + (np.outer(self.pre_tr, pkj_out) - np.outer(grc_out, self.post_tr))

        # 2) CF gating (vector per PKJ if available)
        if cf_signal is None:
            if self._last.get('cf') is not None:
                cf = self._last['cf']           # (N_PKJ,)
            else:
                cf = 1.0                        # scalar gate
        else:
            cf = cf_signal                      # scalar or (N_PKJ,)

        # PATCH: in CerebellarNet.learn(), just before applying to weight:
        with self.torch.no_grad():
            E_t = self.torch.from_numpy(self.E).to(self.dev)     # [N_GRC, N_PKJ]
            if np.isscalar(cf):
                dW = self.cfg.eta * float(cf) * E_t
            else:
                cf_vec = self.torch.from_numpy(np.asarray(cf, dtype=np.float32)).to(self.dev)  # [N_PKJ]
                dW = self.cfg.eta * (E_t * cf_vec)   # broadcast on last dim

            # --- NEW: per-step update clamp (stability) ---
            dW.clamp_(-self.cfg.max_dw_per_step, self.cfg.max_dw_per_step)

            self.grc2pkj.weight += dW.T              # [N_PKJ, N_GRC]
            self.grc2pkj.weight.clamp_(-1.5, 1.5)

            # --- NEW: periodic synaptic scaling (row L2 ≤ 1) ---
            if not hasattr(self, "_scale_k"):
                self._scale_k = 0
            self._scale_k += 1
            if (self._scale_k % self.cfg.scale_every) == 0:
                W = self.grc2pkj.weight
                rownorm = W.norm(p=2, dim=1, keepdim=True).clamp_min(1.0)
                W.mul_(1.0 / rownorm)



def encode_mf_spikes(obs, rate_scale=60.0, dt=DT):
    x = np.asarray(obs, dtype=np.float32)
    rng = float(np.ptp(x))
    x = (x - float(x.min())) / (rng + 1e-6) if rng >= 1e-12 else np.zeros_like(x, dtype=np.float32)
    lam = x * rate_scale * dt
    return (np.random.rand(x.shape[0]) < lam).astype(np.float32)


# 4-edge → 2-hinge torque
def edge4_to_torque2(u4, w=(1.,1.,1.,1.), max_tau=(0.2,0.2)):
    uU, uR, uD, uL = u4
    wU, wR, wD, wL = w
    tau_y = (wU*uU - wD*uD) * max_tau[1]
    tau_x = (wR*uR - wL*uL) * max_tau[0]
    return float(np.clip(tau_x, -max_tau[0], max_tau[0])), \
           float(np.clip(tau_y, -max_tau[1], max_tau[1]))