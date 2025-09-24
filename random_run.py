# run_live.py
import numpy as np
from viewer_utils import LiveViewer
from mujoco_model import Environment, SimulationConfig
# from your_module import Environment, SimulationConfig

cfg = SimulationConfig()
env = Environment(cfg)
obs, info = env.reset()

lv = LiveViewer(env.model, env.data)
T = 1000
for t in range(T):
    # 예시: 랜덤 제어
    action = np.random.rand(4).astype(np.float32)
    obs, reward, terminated, truncated, step_info = env.step(action)

    lv.sync()  # 뷰어 업데이트

    if terminated or truncated:
        break

lv.close()
