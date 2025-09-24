# eval_policy.py
import numpy as np
import torch
from torch.distributions import Bernoulli

def obs_to_tensor(obs_bool_10x10, device):
    x = torch.from_numpy(obs_bool_10x10.astype(np.float32))  # (10,10) {0,1}
    return x.unsqueeze(0).unsqueeze(0).to(device)            # (B=1,C=1,H=10,W=10)

def spikes_to_action(spikes_4):
    return spikes_4.detach().cpu().numpy().astype(np.float32)

@torch.no_grad()
def run_one_episode(env, policy, device, mode="deterministic", thresh=0.5, max_steps=2000):
    """
    mode: 'deterministic' -> sigmoid(logits) >= thresh
          'stochastic'    -> Bernoulli(logits=logits).sample()
    Returns: episode_return, steps, success_flag
    """
    obs, info = env.reset()
    ep_return, steps = 0.0, 0
    success_flag = False

    for t in range(max_steps):
        x = obs_to_tensor(obs, device)
        logits = policy(x)  # (1,4)

        if mode == "stochastic":
            dist = Bernoulli(logits=logits)         # sample {0,1}
            a = dist.sample()[0]                    # (4,)
        else:  # deterministic
            probs = torch.sigmoid(logits)[0]        # (4,)
            a = (probs >= thresh).to(probs.dtype)   # (4,)

        action_vec = spikes_to_action(a)
        obs, reward, terminated, truncated, info = env.step(action_vec)

        ep_return += float(reward)
        steps += 1

        # success heuristic (adapt to your env keys)
        # e.g., you increment env.success_count when within 1 cm of center
        success_flag = success_flag or (getattr(env, "success_count", 0) >= getattr(env, "success_timestep", 1))

        if terminated or truncated:
            break

    return ep_return, steps, success_flag

@torch.no_grad()
def evaluate(env_ctor, policy, device, n_eval_episodes=20, mode="deterministic", thresh=0.5):
    """
    env_ctor: callable that returns a FRESH env (important if stateful)
    """
    returns, lengths, successes = [], [], []
    for _ in range(n_eval_episodes):
        env = env_ctor()             # fresh env instance
        env.reset()                  # reset before eval
        R, L, S = run_one_episode(env, policy, device, mode=mode, thresh=thresh)
        returns.append(R); lengths.append(L); successes.append(1 if S else 0)

    returns = np.asarray(returns, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int32)
    successes = np.asarray(successes, dtype=np.int32)

    report = {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
        "mean_length": float(lengths.mean()),
        "success_rate": float(successes.mean()),
        "episodes": int(n_eval_episodes),
        "mode": mode,
    }
    return report
# EOF