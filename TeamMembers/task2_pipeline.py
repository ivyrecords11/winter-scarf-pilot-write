# task2_pipeline.py
# 3.3%
# 2차 과제 파이프라인 (센서100 + 지연큐 + SNN 학습 + 폐루프 평가 + 가중치 export)
# 실행 예)
#   데이터 생성:  python task2_pipeline.py --make_dataset
#   학습:         python task2_pipeline.py --train
#   평가:         python task2_pipeline.py --eval
#   내보내기:     python task2_pipeline.py --export

import math, random, argparse, os, json
from collections import deque
import numpy as np

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import utils


# ----------------------- 공통 하이퍼파라미터 -----------------------
PLANE_CM = 30.0          # 30 x 30 cm
DT       = 0.01          # 10ms timestep (100Hz)
STEPS_EP = int(8.0/DT)   # 에피소드 길이 8초
G_CM     = 980.0         # 중력 cm/s^2
MU       = 0.25          # 감쇠(구름 마찰 등가)
BOUNCE   = 0.2           # 경계 반발
MAX_TILT = 8.0           # 최대 기울기(deg)
V_SIG_CM = 50000.0       # 신경 전송속도: 500 m/s = 50000 cm/s
NSENS    = 100           # 센서 개수
K_HIST   = 20            # 최근 K프레임
SEED     = 42

DATA_FILE = "data_train.npz"
CKPT_FILE = "snn_ckpt.pth"
EXPORT_NPZ = "snn_export_int8.npz"

rng = np.random.default_rng(SEED)
random.seed(SEED)

# ----------------------- 유틸 -----------------------
def mass_to_radius_cm(mass_g: float) -> float:
    # 지름 ~ m^(1/3) 가정 → 반지름 cm 스케일
    base_r = 0.25  # 1g 기준 반지름 0.25cm (시각화/물리 안정용)
    return base_r * (mass_g ** (1.0/3.0))

def clamp(x, a, b): return a if x < a else b if x > b else x

def hist_to_cop_features(hist_win: np.ndarray, sensors_xy: np.ndarray):
    """
    hist_win: (K, NSENS) 최근 K프레임의 0/1 도착 벡터
    sensors_xy: (NSENS, 2) 각 센서 좌표(cm)
    return: (4,) = [x_hat, y_hat, vx_hat, vy_hat]  (cm, cm/s)
    """
    K, N = hist_win.shape

    # 시간 가중치: (K,1)로 유지해 브로드캐스팅 OK
    w = np.linspace(1.0, 2.0, K, dtype=np.float32).reshape(K, 1)
    W = hist_win * w                                 # (K,NSENS)
    a = W.sum(axis=0, keepdims=True)                 # (1,NSENS)
    total = float(a.sum())

    if total < 1e-6:
        xy_hat = np.array([[0.0, 0.0]], dtype=np.float32)
    else:
        xy_hat = (a @ sensors_xy) / total            # (1,2)

    # 속도 근사: 앞/뒤 절반창의 COP 차이 / 시간
    mid = max(1, K // 2)

    a1 = hist_win[:mid].sum(axis=0, keepdims=True)   # (1,NSENS)
    t1 = float(a1.sum())
    xy1 = (a1 @ sensors_xy) / (t1 + 1e-6)

    a2 = hist_win[mid:].sum(axis=0, keepdims=True)   # (1,NSENS)
    t2 = float(a2.sum())
    xy2 = (a2 @ sensors_xy) / (t2 + 1e-6)

    dt = (K - mid) * DT
    if dt <= 0: dt = DT
    v_hat = (xy2 - xy1) / dt                         # (1,2)

    return np.hstack([xy_hat.ravel(), v_hat.ravel()]).astype(np.float32)


# ----------------------- 센서 배치 -----------------------
def make_sensors(mode="grid-jitter"):
    half = PLANE_CM/2
    if mode == "grid-jitter":
        side = int(math.sqrt(NSENS))
        assert side*side == NSENS, "NSENS는 제곱수가 편합니다. (100 권장)"
        gx = np.linspace(-half, half, side)
        gy = np.linspace(-half, half, side)
        gxx, gyy = np.meshgrid(gx, gy)
        base = np.column_stack([gxx.ravel(), gyy.ravel()])
        jitter = rng.normal(0, 0.3, size=base.shape)  # 3mm 표준편차
        pos = base + jitter
    elif mode == "random":
        pos = rng.uniform(-half, half, size=(NSENS,2))
    else:
        raise ValueError("mode should be grid-jitter or random")
    return pos.astype(np.float32)

# ----------------------- 환경/시뮬 -----------------------
class DelayQueues:
    """정수 타임스텝 지연 링큐"""
    def __init__(self, max_steps):
        self.Q = [deque() for _ in range(max_steps)]
        self.idx = 0
        self.max_steps = max_steps

    def enqueue(self, sensor_idx, delay_steps):
        s = (self.idx + int(delay_steps)) % self.max_steps
        self.Q[s].append(sensor_idx)

    def tick_and_pop(self):
        out = list(self.Q[self.idx])
        self.Q[self.idx].clear()
        self.idx = (self.idx + 1) % self.max_steps
        return out

class SimEnv:
    def __init__(self, sensor_mode="grid-jitter"):
        self.sensors = make_sensors(sensor_mode)
        self.controller_xy = np.array([0.0,0.0], np.float32)  # 제어 뉴런 허브 중심
        # 최대 지연 스텝: 대각선 거리 / 속도
        max_delay_s = (math.sqrt(2)*(PLANE_CM/2)) / V_SIG_CM
        self.max_delay_steps = max(2, int(math.ceil(max_delay_s/DT))+2)
        self.queues = DelayQueues(self.max_delay_steps)
        self.mass_g = 10.0
        self.r_cm   = mass_to_radius_cm(self.mass_g)
        self.reset()

    def reset(self, random_mass=True, random_pos=True, maybe_relayout=True):
        if random_mass:
            self.mass_g = float(rng.uniform(1.0, 30.0))
            self.r_cm   = mass_to_radius_cm(self.mass_g)
        if random_pos:
            half = PLANE_CM/2 - self.r_cm - 0.5
            self.pos = np.array([rng.uniform(-half, half), rng.uniform(-half, half)], np.float32)
        self.vel = np.array([0.0,0.0], np.float32)
        self.t = 0
        if maybe_relayout and (rng.random() < 0.25):
            self.sensors = make_sensors("random")

        # 센서별 지연(고정)
        d = np.linalg.norm(self.sensors - self.controller_xy, axis=1)  # cm
        self.delay_steps = np.ceil((d / V_SIG_CM) / DT).astype(np.int32)

    # ---- 센서 모델 ----
    def sensor_rates_hz(self):
        # 거리 기반 가우시안 + 무게 스케일 (압력∝무게)
        # 공의 중심과 센서 거리
        d = np.linalg.norm(self.sensors - self.pos, axis=1)
        sigma = 4.0  # cm
        base = np.exp(-(d**2)/(2*sigma**2))
        # 멀어지면 0 (cutoff)
        base[d > 3.0*sigma] = 0.0
        # 무게 스케일
        rate = base * (5.0 + 0.6*self.mass_g)  # 1g~30g → 대략 5~23Hz
        return rate.astype(np.float32)

    def emit_and_enqueue(self):
        # 포아송 발생 → 큐 삽입
        rate = self.sensor_rates_hz()          # [100]
        p = 1 - np.exp(-rate*DT)               # 각 센서에서 이번 스텝 발화 확률
        fires = (rng.random(NSENS) < p)
        idxs = np.nonzero(fires)[0]
        for i in idxs:
            self.queues.enqueue(i, self.delay_steps[i])

    def dequeue_arrivals(self):
        arr = self.queues.tick_and_pop()
        return arr

    # ---- 디코더: 센서 도착 카운트(이번 스텝) → 100-d 벡터 ----
    def arrivals_to_vec(self, arrivals):
        vec = np.zeros(NSENS, np.float32)
        if len(arrivals):
            vec[arrivals] = 1.0
        return vec

    # ---- 교사(Teacher) PD 제어 ----
    def teacher_pd(self, kp=1.2, kd=0.5):
        # 중앙으로 보내기: 기울기는 (음의) 위치/속도에 비례
        # 반대 매핑: 왼쪽이면 오른쪽으로 기울인다 → tilt_x = +kp*(-x) - kd*vx
        x,y = float(self.pos[0]), float(self.pos[1])
        vx,vy = float(self.vel[0]), float(self.vel[1])
        tilt_x = clamp( kp*(-x) - kd*vx, -MAX_TILT, MAX_TILT )
        tilt_y = clamp( kp*(-y) - kd*vy, -MAX_TILT, MAX_TILT )
        return np.array([tilt_x, tilt_y], np.float32)

    # ---- 물리 ----
    def step_physics(self, tilt):
        tx, ty = float(tilt[0]), float(tilt[1])
        ax = G_CM * math.sin(math.radians(tx))
        ay = G_CM * math.sin(math.radians(ty))
        # 감쇠 포함
        self.vel[0] += (ax - MU*self.vel[0]) * DT
        self.vel[1] += (ay - MU*self.vel[1]) * DT
        self.pos += self.vel * DT

        # 경계
        half = PLANE_CM/2 - self.r_cm
        if self.pos[0] < -half:
            self.pos[0] = -half; self.vel[0] *= -BOUNCE
        if self.pos[0] >  half:
            self.pos[0] =  half; self.vel[0] *= -BOUNCE
        if self.pos[1] < -half:
            self.pos[1] = -half; self.vel[1] *= -BOUNCE
        if self.pos[1] >  half:
            self.pos[1] =  half; self.vel[1] *= -BOUNCE
        self.t += 1

    def settled(self, radius_cm=1.0, window_steps=int(0.5/DT)):
        # 최근 window 동안 중심 반경 내에 있으면 정착
        return (np.linalg.norm(self.pos) < radius_cm)

# ----------------------- 데이터 생성 -----------------------
def make_dataset(episodes=200, sensor_mode="grid-jitter", save=DATA_FILE):
    env = SimEnv(sensor_mode)
    Xs, Ys = [], []
    hist = np.zeros((K_HIST, NSENS), np.float32)

    for ep in range(episodes):
        env.reset(random_mass=True, random_pos=True, maybe_relayout=True)
        hist[:] = 0.0
        for t in range(STEPS_EP):
            env.emit_and_enqueue()
            arrivals = env.dequeue_arrivals()
            vec = env.arrivals_to_vec(arrivals)
            hist = np.roll(hist, -1, axis=0); hist[-1] = vec

            # 교사 제어 → 물리 적용
            u = env.teacher_pd()
            env.step_physics(u)

            if t >= K_HIST:
                cop4 = hist_to_cop_features(hist, env.sensors)  # (4,)
                Xs.append(np.concatenate([hist.reshape(-1), cop4], axis=0))
                Ys.append(u)

    X = np.stack(Xs).astype(np.float32)
    y = np.stack(Ys).astype(np.float32)
    meta = dict(K=K_HIST, NSENS=NSENS, DT=DT, MAX_TILT=MAX_TILT)
    np.savez_compressed(save, X=X, y=y, meta=meta)
    print(f"[make_dataset] saved {save}  X:{X.shape}  y:{y.shape}")

# ----------------------- SNN 모델/학습 -----------------------
class SNNReg(nn.Module):
    def __init__(self, D, H=128):  # H=128 권장
        super().__init__()
        self.fc1  = nn.Linear(D, H)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan(), init_hidden=True)

        self.fc2  = nn.Linear(H, H//2)  # 🔁 중간층 추가
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.atan(), init_hidden=True)

        self.head = nn.Linear(H//2, 2)
        self.act  = nn.Tanh()  # [-1,1]

    def forward(self, x):
        h1   = self.fc1(x)
        o1   = self.lif1(h1)
        mem1 = o1[1] if isinstance(o1, (tuple, list)) and len(o1)>=2 else o1

        h2   = self.fc2(mem1)
        o2   = self.lif2(h2)
        mem2 = o2[1] if isinstance(o2, (tuple, list)) and len(o2)>=2 else o2

        u = self.head(mem2)
        return MAX_TILT * self.act(u)  # 각도 한계로 직접 제한



def train_snn(data_file=DATA_FILE, ckpt=CKPT_FILE,
              epochs=20, lr=5e-4, hidden=128, batch=2048):  # ← epochs=20
    npz = np.load(data_file, allow_pickle=True)
    X = torch.tensor(npz["X"], dtype=torch.float32)
    y = torch.tensor(npz["y"], dtype=torch.float32)
    D = X.shape[1]

    X_mean = X.mean(0, keepdim=True)
    X_std  = X.std(0, keepdim=True) + 1e-6
    Xn = (X - X_mean) / X_std

    model = SNNReg(D, H=hidden).to(device)  # ← 모델을 GPU로
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 큰 배치에서도 빠르게: matmul 최적화
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    idx = torch.randperm(Xn.shape[0])
    for ep in range(epochs):
        ep_loss = 0.0
        for i in range(0, len(idx), batch):
            sl = idx[i:i+batch]
            xb = Xn[sl].to(device, non_blocking=True)  # ← 배치만 GPU로
            yb = y[sl].to(device, non_blocking=True)

            opt.zero_grad()
            utils.reset(model)
            pred = model(xb)
            with torch.no_grad():
                w = torch.linalg.vector_norm(yb, dim=1) / MAX_TILT
                w = torch.clamp(w, 0, 1)
                w = 0.2 + 0.8 * w
            loss = torch.mean(w * torch.mean((pred - yb)**2, dim=1))
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(sl)

        ep_loss /= len(idx)
        print(f"[train] epoch {ep+1}/{epochs}  loss {ep_loss:.6f}")

    # 저장은 CPU/GPU 상관없음
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": D,
        "hidden": hidden,
        "X_mean": X_mean.cpu().numpy(),
        "X_std":  X_std.cpu().numpy(),
    }, ckpt)
    print(f"[train] saved {ckpt}")

# ----------------------- 폐루프 평가 -----------------------
#
def eval_teacher(episodes=10, sensor_mode="grid-jitter"):
    env = SimEnv(sensor_mode)
    settle = []
    for ep in range(episodes):
        env.reset(True, True, True)
        for t in range(STEPS_EP):
            u = env.teacher_pd()
            env.step_physics(u)
            if env.settled(radius_cm=1.0) and t*DT>0.8:
                settle.append(t*DT); break
        else:
            settle.append(np.nan)
    print("teacher median=", np.nanmedian(settle), "fail%=", float(np.mean(np.isnan(settle))*100))
#

def eval_closedloop(ckpt=CKPT_FILE, episodes=30, sensor_mode="grid-jitter"):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    print("ckpt D =", ck["input_dim"])
    print("eval feature options: base=2000, base+cop4=2004")
    D = ck["input_dim"]; H = ck["hidden"]
    model = SNNReg(D, H).to(device).eval()     # ← 모델 GPU로
    model.load_state_dict(ck["state_dict"])
    X_mean = torch.tensor(ck["X_mean"]).to(device)
    X_std  = torch.tensor(ck["X_std"]).to(device)

    env = SimEnv(sensor_mode)
    hist = np.zeros((K_HIST, NSENS), np.float32)
    settle_times = []

    for ep in range(episodes):
        utils.reset(model)
        env.reset(True, True, False)
        hist[:] = 0.0
        settled_at = None

        beta = 0.0
        prev_u = np.zeros(2, dtype=np.float32)

        # --- 추가: 성공 판정 상태변수 ---
        R = 1.8                         # 성공 반경(cm) 완화
        REQ = int(0.4 / DT)             # 0.4초(연속 스텝 수)
        in_radius_count = 0

        for t in range(STEPS_EP):
            env.emit_and_enqueue()
            arr = env.dequeue_arrivals()
            vec = env.arrivals_to_vec(arr)
            hist = np.roll(hist, -1, axis=0); hist[-1] = vec

            if t >= K_HIST:
                base = hist.reshape(-1)
                want_D = X_mean.shape[1] if X_mean.ndim == 2 else X_mean.numel()
                if base.size + 4 == want_D:
                    cop4 = hist_to_cop_features(hist, env.sensors)
                    feat = np.concatenate([base, cop4], axis=0)
                else:
                    feat = base
                x = torch.from_numpy(feat).float().to(device).unsqueeze(0)
                x = (x - X_mean) / X_std
                u_model = model(x).detach().cpu().numpy().ravel().astype(np.float32)
            else:
                u_model = np.array([0.0, 0.0], np.float32)

            # ---- COP 기반 PD 보조제어 ----
            cop4 = hist_to_cop_features(hist, env.sensors)
            x_hat, y_hat, vx_hat, vy_hat = cop4
            r = float(np.hypot(x_hat, y_hat))

            # --- COP 기반 PD 이전/근처에 교체 ---
            # r-적응형 속도 클립 (near: ~5cm/s, far(>=12): ~12cm/s)
            v_clip = float(np.clip(5.0 + 0.6*r, 5.0, 12.0))
            vx_hat = float(np.clip(vx_hat, -v_clip, v_clip))
            vy_hat = float(np.clip(vy_hat, -v_clip, v_clip))

            # 가변 게인 (더 낮고 부드럽게)
            kp_eff = float(np.clip(1.3 + 0.055*r, 1.3, 2.4))
            kd_eff = float(np.clip(1.05 + 0.050*r, 1.05, 1.9))

            # r, x_hat, y_hat, vx_hat, vy_hat 계산된 이후 ~ u_aux 만들기 전에/후에 OK
            # 방사속도: v_r = (v · r_hat) ~= (vx*x + vy*y) / (r + eps)
            eps = 1e-6
            v_r = float((vx_hat * x_hat + vy_hat * y_hat) / (r + eps))  # cm/s, +면 바깥으로 가는 중

            u_aux = np.array([
                np.clip(kp_eff*(-x_hat) - kd_eff*vx_hat, -MAX_TILT, MAX_TILT),
                np.clip(kp_eff*(-y_hat) - kd_eff*vy_hat, -MAX_TILT, MAX_TILT),
            ], dtype=np.float32)

            # ---- 보조→모델 전환: 시간 + 거리 기반, 상태적(EMA) ----
            tsec = t * DT
            tau_start, tau_full = 0.6, 2.0
            beta_time = 0.0 if tsec <= tau_start else min(1.0, (tsec - tau_start)/(tau_full - tau_start))

            beta_dist_target = float(np.clip(0.95 - 0.09*r, 0.25, 0.95))
            if tsec >= 1.2: beta_dist_target = max(beta_dist_target, 0.35)
            if tsec >= 1.5: beta_dist_target = max(beta_dist_target, 0.40)
            if tsec >= 3.0: beta_dist_target = max(beta_dist_target, 0.50)

            # ★ r-트렌드 백오프: 바깥으로 가는 속도가 빠를수록 모델믹싱 감소
            k_backoff = 0.04  # (cm/s)^-1 정도로 시작해서 0.03~0.06 범위 튜닝
            beta_dist_target -= k_backoff * max(0.0, v_r)
            beta_dist_target = float(np.clip(beta_dist_target, 0.25, 0.95))

            beta_target = min(beta_time, beta_dist_target)
            beta = float(np.clip(beta + 0.20*(beta_target - beta), 0.0, 1.0))
            
           
            # ---- 제어 합성
            u = (1.0 - beta)*u_aux + beta*u_model

            # 벽 근처 부스트
            half = PLANE_CM/2 - env.r_cm
            margin_x = half - abs(env.pos[0])
            margin_y = half - abs(env.pos[1])
            boost_scale = (1.0 - beta)  # ★ 추가: 모델믹싱이 클수록 벽 부스트 약화

            if margin_x < 6.0:
                u[0] += boost_scale * float(-0.5 * (1.0 - margin_x/6.0) * np.sign(env.pos[0]))
            if margin_y < 6.0:
                u[1] += boost_scale * float(-0.5 * (1.0 - margin_y/6.0) * np.sign(env.pos[1]))
            
            # ✅ 최종 클램프 먼저
            u[0] = max(-MAX_TILT, min(MAX_TILT, u[0]))
            u[1] = max(-MAX_TILT, min(MAX_TILT, u[1]))

            # ... 벽 부스트 + 축별 클램프 이후
            umax_final = 9.5
            unorm = float(np.linalg.norm(u)) + 1e-6
            if unorm > umax_final:
                u *= (umax_final / unorm)

            r_in, r_out = 1.2, 1.6
            if r < r_in:
                scale = (r / r_in) ** 0.7
                u *= scale
            # r이 r_out 넘을 때만 스케일 해제 → 자연스런 들썩임 억제


            # ✅ r-적응형 rate limit (멀리 클수록 많이, 가까이 작게)
            max_d_far = 1.0  # deg/step @100Hz → 100°/s
            max_d_near = 0.6
            alpha = float(np.clip(r/10.0, 0.0, 1.0))   # r≈0→0, r≥10→1
            max_d = max_d_near*(1.0 - alpha) + max_d_far*alpha

            du = np.clip(u - prev_u, -max_d, +max_d)
            u = prev_u + du
            prev_u = u.copy()


            if t % 50 == 0 and t >= K_HIST:
                print(f"t={t:4d} r={r:.2f} beta_time={beta_time:.2f} "
                    f"beta_dist={beta_dist_target:.2f} beta={beta:.2f} "
                    f"|u_aux|={np.linalg.norm(u_aux):.2f} |u_model|={np.linalg.norm(u_model):.2f} |u|={np.linalg.norm(u):.2f}")



            env.step_physics(u)

            # --- 연속 체류 성공 판정 ---
            if np.linalg.norm(env.pos) < R:
                in_radius_count += 1
            else:
                in_radius_count = 0

            if (t*DT > 0.8) and (in_radius_count >= REQ):
                settled_at = t*DT - REQ*DT  # 처음 반경 진입 시점으로 보정
                break


        settle_times.append(settled_at if settled_at is not None else np.nan)

    median = np.nanmedian(settle_times)
    p90    = np.nanpercentile(settle_times, 90)
    fail   = float(np.mean(np.isnan(settle_times))*100.0)
    print(f"[eval] settle median={median:.2f}s  p90={p90:.2f}s  fail%={fail:.1f}")
    return settle_times

# ----------------------- 가중치 export (int8) -----------------------
def quant_int8(W: np.ndarray):
    s = np.max(np.abs(W))
    scale = (s/127.0) if s>0 else 1.0
    Wq = np.clip(np.round(W/scale), -127, 127).astype(np.int8)
    return Wq, np.array([scale], np.float32)


def export_int8(ckpt=CKPT_FILE, out=EXPORT_NPZ):
    ck = torch.load(ckpt, map_location="cpu",weights_only=False)
    sd = ck["state_dict"]
    W1 = sd["fc1.weight"].cpu().numpy(); b1 = sd["fc1.bias"].cpu().numpy()
    W2 = sd["fc2.weight"].cpu().numpy(); b2 = sd["fc2.bias"].cpu().numpy()
    W1_q, S_w1 = quant_int8(W1); b1_q, S_b1 = quant_int8(b1)
    W2_q, S_w2 = quant_int8(W2); b2_q, S_b2 = quant_int8(b2)

    np.savez_compressed(out,
        W1_q=W1_q, S_w1=S_w1, b1_q=b1_q, S_b1=S_b1,
        W2_q=W2_q, S_w2=S_w2, b2_q=b2_q, S_b2=S_b2,
        X_mean=ck["X_mean"].astype(np.float32),
        X_std =ck["X_std"].astype(np.float32),
        meta=dict(D=ck["input_dim"], H=ck["hidden"], K=K_HIST, NSENS=NSENS, DT=DT))
    print(f"[export] saved {out}")

# ----------------------- 메인 -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_dataset", action="store_true", help="데이터 생성")
    ap.add_argument("--train",        action="store_true", help="SNN 학습")
    ap.add_argument("--eval",         action="store_true", help="폐루프 평가")
    ap.add_argument("--export",       action="store_true", help="가중치 export(int8)")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--sensor_mode", type=str, default="grid-jitter", choices=["grid-jitter","random"])
    args, unknown = ap.parse_known_args()

    if args.make_dataset:
        make_dataset(episodes=args.episodes, sensor_mode=args.sensor_mode)
    if args.train:
        train_snn()
    if args.eval:
        eval_closedloop(sensor_mode=args.sensor_mode)
    if args.export:
        export_int8()

if __name__ == "__main__":
    main()
