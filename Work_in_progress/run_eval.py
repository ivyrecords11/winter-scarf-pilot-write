# run_eval.py
import torch
from mujoco_model import Environment, SimulationConfig
from policy import SpikePolicy
from eval_policy import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = SimulationConfig()
policy = SpikePolicy(hidden=64).to(device)

def make_env():
    # fresh env each episode (mirrors SB3’s recommendation)
    return Environment(cfg)

# deterministic eval (thresholded spikes)
rep_det = evaluate(make_env, policy, device, n_eval_episodes=20, mode="deterministic", thresh=0.5)
print("[EVAL deterministic]", rep_det)

# stochastic eval (sample Bernoulli from logits)
rep_sto = evaluate(make_env, policy, device, n_eval_episodes=20, mode="stochastic")
print("[EVAL stochastic]", rep_sto)
