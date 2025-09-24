# local_control.py
import time, numpy as np, mujoco
import mujoco.viewer
import torch

# 1) 모델 로드
m = mujoco.MjModel.from_xml_path("tiltPlate.xml")
d = mujoco.MjData(m)


# ==== 3) Discrete actions {0,1}^4 ====
N_BINS = 2
BINS   = np.array([0.0, 1.0], dtype=np.float32)
def idx_to_u4(idx, n_bins=N_BINS, bins=BINS):
    u = np.empty(4, dtype=np.float32)
    for k in range(4):
        u[k] = bins[idx % n_bins]; idx //= n_bins
    return u  # [U,R,D,L]

# (선택) SNN 또는 RLlib 로더 붙이기 -----------------
USE_SNN = True
'''
import torch, pprint, sys
    p='policy_weights.pt'
    try:
        ckpt=torch.load(p,map_location='cpu', weights_only=False)
    except Exception as e:
        print('load error',e); sys.exit(1)
    print('type:',type(ckpt))
    if isinstance(ckpt, dict):
        print('keys:')
        pprint.pprint(list(ckpt.keys()))
        # show short repr of values
        for k in ckpt:
            v = ckpt[k]
            print('-',k,'->',type(v))
            if hasattr(v,'keys'):
                try:
                    print('  subkeys:', list(v.keys())[:10])
                except Exception:
                    pass
    else:
        print('repr of object:',repr(ckpt)[:500])'''

'''if USE_SNN:
    # SNN 복구
    import torch
    from DLModules import CerebellarNet, SimulationConfig, encode_mf_spikes, edge4_to_torque2  # 네 코드
    ckpt = torch.load("policy_weights.pt", map_location="cpu", weights_only=False)
    cfg = SimulationConfig( 
    n_mf=108, n_grc=512, n_pkj=8, n_motor=4,
    tau_pre_ms=5.0, tau_post_ms=5.0, tau_e_ms=50.0,
    dt_ms=0.1, eta=1e-3, max_dw_per_step=1e-3, scale_every=100,
    use_mf_collat=True, mf_collat_gain=0.3)
    
    snn = CerebellarNet(cfg)
    snn.mf2grc.load_state_dict(ckpt)
    snn.grc2pkg.load_state_dict(ckpt)
    snn.eval()
else:
    # RLlib 복구
    from ray.rllib.algorithms.algorithm import Algorithm
    restored = Algorithm.from_checkpoint("path/to/checkpoint_dir")  # 권장 API
    # 새 스택에선 예제/마이그 가이드 흐름을 따르세요(일부 구 API는 폐지). :contentReference[oaicite:6]{index=6}'''
from ray.rllib.algorithms.algorithm import Algorithm
algo = Algorithm.from_checkpoint("./ckpt_export")
# 추론/재학습 OK


# 2) 뷰어 실행(비차단)
with mujoco.viewer.launch_passive(m, d) as v:  # 윈도우 닫히면 자동 종료
    start = time.time()
    while v.is_running() and (time.time() - start < 60):
        step_start = time.time()

        # ---- 관측 만들기 (네 env의 obs와 유사하게 구성) ----
        bx, by = d.qpos[0], d.qpos[1]
        vx, vy = d.qvel[0], d.qvel[1]
        obs = np.array([bx, by, vx, vy], dtype=np.float32)  # 필요시 확장

        # ---- 제어: SNN 또는 RLlib ----
        if USE_SNN:
            mf_spk = encode_mf_spikes(obs, rate_scale=60.0, dt=m.opt.timestep)
            mo_spk = snn.forward(mf_spk, cf_spikes=None)      # (4,) 0/1
            u4 = mo_spk.astype(np.float32)
        else:
            # RLlib: 새 API에서 단일 액션 추출은 예제가 가장 안전.
            # 여기선 의사코드 (알고리즘/정책 API deprecations 주의)
            action = restored.predict(obs)  # 예: 실제 구현은 docs/examples 참고
            # {0,1}^4 디코딩
            u4 = idx_to_u4(int(action))     # 네가 쓰는 디코더

        # ---- 토크 적용 & 스텝 ----
        tau_x = (u4[1] - u4[3]) * 0.2
        tau_y = (u4[0] - u4[2]) * 0.2
        d.ctrl[0] = float(tau_x)
        d.ctrl[1] = float(tau_y)

        mujoco.mj_step(m, d)

        # (옵션) 뷰옵션 변경은 lock 내에서
        with v.lock():
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2 == 0)

        v.sync()  # GUI ↔ physics 동기화

        # 타임킵(실시간에 가깝게)
        sleep = m.opt.timestep - (time.time() - step_start)
        if sleep > 0: time.sleep(sleep)
