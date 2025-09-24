# run_live.py
import os
import numpy as np
import torch
from torch.distributions import Bernoulli

import mujoco as mj
import mujoco.viewer as viewer
from mujoco_model import Environment, SimulationConfig
#from SNNModels 
from policy import SpikePolicy  # your simple ANN: 10x10 -> 4 logits

# --- config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_STOCHASTIC = True       # True: sample Bernoulli; False: deterministic threshold
DET_THRESHOLD = 0.5         # used only when USE_STOCHASTIC=False
CHECKPOINT = "./checkpoints/ep110"  # optional: path to .pt / .pth

def obs_to_tensor(obs):
    """obs: (10,10) np.bool/float -> (1,1,10,10) float32 on DEVICE"""
    if obs.ndim == 1 and obs.size == 100:
        obs = obs.reshape(10, 10)
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return x.to(DEVICE)

def spikes_to_action(spikes_or_probs):
    """tensor (4,) in {0,1} or [0,1] -> numpy float32 [XP,XN,YP,YN]"""
    return spikes_or_probs.detach().cpu().numpy().astype(np.float32)

def main():
    # --- env ---
    cfg = SimulationConfig()
    env = Environment(cfg)
    obs, info = env.reset()

    # --- policy ---
    policy = SpikePolicy(hidden=64).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        policy.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        print(f"[run_live] loaded weights: {CHECKPOINT}")
    policy.eval()  # eval mode (dropout/bn)  ← not the same as no_grad :contentReference[oaicite:3]{index=3}

    # --- viewer (passive) ---
    v = viewer.launch_passive(env.model, env.data)  # must call v.sync() each step :contentReference[oaicite:4]{index=4}

    T = 10000
    with torch.no_grad():  # disable autograd for fast eval :contentReference[oaicite:5]{index=5}
        for t in range(T):
            # 1) policy forward
            x = obs_to_tensor(obs)                 # (1,1,10,10)
            logits = policy(x)[0]                  # (4,)
            if USE_STOCHASTIC:
                # Bernoulli with logits -> {0,1} spikes
                dist = Bernoulli(logits=logits)    # params in logit space :contentReference[oaicite:6]{index=6}
                spikes = dist.sample()             # (4,) in {0,1}
                action_vec = spikes_to_action(spikes)
            else:
                probs = torch.sigmoid(logits)      # (4,) in [0,1]
                spikes = (probs >= DET_THRESHOLD).to(probs.dtype)
                action_vec = spikes_to_action(spikes)

            # 2) env step
            obs, reward, terminated, truncated, info = env.step(action_vec)

            # 3) (optional) log a bit
            if (t % 50) == 0:
                pos = env.data.body("ball").xpos.copy()  # copy() is recommended :contentReference[oaicite:7]{index=7}
                print(f"t={t:04d} | pos=({pos[0]:+.3f},{pos[1]:+.3f}) | r={reward:+.2f}")

            # 4) render
            v.sync()  # update viewer every step (passive viewer API) :contentReference[oaicite:8]{index=8}

            if terminated or truncated:
                print(f"[END] terminated={terminated}, truncated={truncated} at t={t}")
                break

    v.close()

if __name__ == "__main__":
    main()
