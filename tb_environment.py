from mujoco_model import Environment, SimulationConfig 
import mujoco as mj
import numpy as np
import sys

def run_smoke_test(Environment):
    # 패치 적용
    Env = Environment

    cfg = SimulationConfig()
    env = Env(cfg)

    # reset: 관측/정보 확인
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray), "reset() observation must be numpy array"
    assert obs.shape == (100,), f"reset obs shape expected (100,), got {obs.shape}"
    assert "ball_pos" in info and "ball_mass" in info

    print("[OK] reset(): obs shape=", obs.shape, " ball_pos=", info["ball_pos"])

    # MuJoCo 객체 유효성
    assert isinstance(env.model, mj.MjModel)
    assert isinstance(env.data, mj.MjData)

    # 10 step 랜덤 액션 실행
    steps = 10
    for t in range(steps):
        action = np.random.rand(4).astype(np.float32)  # [XP,XN,YP,YN] in [0,1]
        out = env.step(action)
        assert len(out) == 5, "step() must return 5-tuple"
        obs, reward, terminated, truncated, step_info = out

        # step()의 obs는 (10,10)로 작성되어 있음
        assert isinstance(obs, np.ndarray) and obs.shape == (10, 10), f"step obs shape expected (10,10), got {obs.shape}"
        assert np.isfinite(reward), "reward must be finite"

        # MuJoCo 상태 점검: 공 위치/속도 유효성
        pos = env.data.body("ball").xpos.copy()
        assert pos.shape == (3,), f"ball xpos shape expected (3,), got {pos.shape}"                                           
        assert np.all(np.isfinite(pos))

        # step_info 필드 존재 확인
        for k in ["ball_x", "ball_y", "plate_angles_rad", "step_count"]:
            assert k in step_info, f"step_info missing key: {k}"

        print(f"t={t:02d} | reward={reward:+.3f} | pos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
              f"| term={terminated} trunc={truncated}")

        if terminated or truncated:
            print("[INFO] Episode ended early: terminated =", terminated, ", truncated =", truncated)
            break

    print("[OK] smoke test completed.")

if __name__ == "__main__":
      # 사용자가 작성한 Environment 클래스를 여기에 import 해두었다고 가정
    # from your_module import Environment
    # 데모를 위해, 같은 파일/세션에 Environment가 정의되어 있다고 가정합니다.
    # 만약 외부 모듈이라면 위 import 주석을 해제하고 아래 줄은 제거하세요.
    try:
        Environment  # noqa
    except NameError:
        print("ERROR: Please import your Environment class at the top of this file.")
        sys.exit(1)

    run_smoke_test(Environment)