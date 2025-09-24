# run_live.py
import numpy as np
import mujoco as mj
from viewer_utils import LiveViewer
from mujoco_model import Environment, SimulationConfig

# --- 간단 이산 PID (anti-windup: clamped integral + output saturation) ---
class PID:
    def __init__(self, Kp, Ki, Kd, dt, u_min=-1.0, u_max=+1.0, d_tau=0.02):
        """
        Kp,Ki,Kd : PID 이득
        dt       : 샘플 주기 (env.dt와 동일하게)
        u_min/max: 출력 포화 한계 (여기선 액션 차(difference) 범위 [-1,1]로 설정)
        d_tau    : D항 저역통과 시정수(초), 미소 진동/노이즈 완화
        """
        self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)
        self.dt = float(dt)
        self.u_min, self.u_max = float(u_min), float(u_max)
        # 상태
        self.integral = 0.0
        self.prev_err = 0.0
        self.d_state = 0.0  # 1st-order filtered derivative
        self.alpha = d_tau / (d_tau + self.dt) if d_tau > 0 else 0.0

    def reset(self, e0=0.0):
        self.integral = 0.0
        self.prev_err = float(e0)
        self.d_state = 0.0

    def update(self, err):
        # P
        p = self.Kp * err
        # I (clamped anti-windup)
        self.integral += err * self.dt
        # D (filtered on error)
        de = (err - self.prev_err) / self.dt
        self.d_state = self.alpha * self.d_state + (1.0 - self.alpha) * de
        d = self.Kd * self.d_state

        u_unsat = p + self.Ki * self.integral + d
        # 출력 포화
        u = max(self.u_min, min(self.u_max, u_unsat))

        # 간단 anti-windup: 포화 시 적분을 한 스텝 되돌림
        if u != u_unsat:
            # 되돌리기: integral -= (u_unsat - u)/Ki  (Ki=0이면 스킵)
            if self.Ki > 0:
                self.integral -= (u_unsat - u) / self.Ki

        self.prev_err = err
        return u

def diff_to_act_pair(s):
    """
    s in [-1, 1] 을 환경 액션 (pos, neg)로 사상.
    env.step 내부에서 data.ctrl = (pos - neg) * 0.2 로 쓰이므로
    여기선 단순 ReLU 분해가 이해하기 쉽다.
    """
    pos = float(max(s, 0.0))
    neg = float(max(-s, 0.0))
    # 안전 클램프
    return min(pos, 1.0), min(neg, 1.0)

# ---- env 생성 ----
cfg = SimulationConfig()
env = Environment(cfg)
obs, info = env.reset()

lv = LiveViewer(env.model, env.data)

# ---- PID 준비: 목표는 공 (x,y) → (0,0) ----
dt = env.dt
# 시작 이득(예시): P 위주, I는 작게, D로 overshoot 완화
pid_x = PID(Kp=100.0, Ki=2.0, Kd=10.0, dt=dt, u_min=-1.0, u_max=+1.0, d_tau=0.05)
pid_y = PID(Kp=100.0, Ki=2.0, Kd=10.0, dt=dt, u_min=-1.0, u_max=+1.0, d_tau=0.05)

T = 100000  # 스텝 수
for t in range(T):
    # 현재 공 위치 (월드 좌표)
    pos = env.data.body("ball").xpos.copy()  # [x,y,z]
    err_x = 0.0 - float(pos[0])
    err_y = 0.0 - float(pos[1])

    # PID 출력: s_x, s_y ∈ [-1,1]  (env.step에서 0.2배 토크 스케일 적용)
    s_x = pid_x.update(err_x)
    s_y = pid_y.update(err_y)

    # 액션 포맷 [XP, XN, YP, YN] (각 [0,1])
    ax_pos, ax_neg = diff_to_act_pair(s_x)
    ay_pos, ay_neg = diff_to_act_pair(s_y)
    action = np.array([ax_pos, ax_neg, ay_pos, ay_neg], dtype=np.float32)

    obs, reward, terminated, truncated, step_info = env.step(action)

    # 렌더
    lv.sync()

    if (t % 50) == 0:
        print(f"t={t:04d} | pos=({pos[0]:+.3f},{pos[1]:+.3f}) | "
              f"act=({ax_pos:.2f},{ax_neg:.2f}; {ay_pos:.2f},{ay_neg:.2f}) | r={reward:+.2f}")

    if terminated or truncated:
        print(f"[END] terminated={terminated}, truncated={truncated} at t={t}")
        break

lv.close()
# EOF