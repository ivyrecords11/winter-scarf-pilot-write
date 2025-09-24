# run_rl.py (live viewer 루프를 대체하는 간단 실행/훈련 예시)
import numpy as np
import torch
from torch.distributions import Bernoulli
from mujoco_model import Environment, SimulationConfig
# from SNNModels import CerebellarNet
from policy import SpikePolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = SimulationConfig()
env = Environment(cfg)

policy = SpikePolicy(hidden=64).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)

def obs_to_tensor(obs_bool_10x10):
    x = torch.from_numpy(obs_bool_10x10.astype(np.float32))  # (10,10) {0,1}
    x = x.unsqueeze(0).unsqueeze(0)  # (B=1,C=1,H=10,W=10)
    return x.to(device)

def spikes_to_action(spikes_4):
    # spikes_4: tensor shape (4,) in {0,1}
    a = spikes_4.detach().cpu().numpy().astype(np.float32)   # [XP, XN, YP, YN]
    return a

# ----------- REINFORCE (한 에피소드) -----------
def run_episode_train(max_steps=10000, gamma=0.99, entropy_coef=1e-3):
    obs, info = env.reset()
    logps, rewards, entropies = [], [], []

    for t in range(max_steps):
        x = obs_to_tensor(obs)                    # (1,1,10,10)

        logits = policy(x)                        # (1,4)
        dist   = Bernoulli(logits=logits)         # 파라미터: 로지트 사용 가능. 샘플은 {0,1} :contentReference[oaicite:1]{index=1}
        a      = dist.sample()                    # (1,4) Bernoulli 샘플 {0,1}
        logp   = dist.log_prob(a).sum(dim=1)      # 다액션 합 로그확률 (1,)
        entropy= dist.entropy().sum(dim=1)        # 엔트로피 보너스

        action_vec = spikes_to_action(a[0])       # numpy 4-vector
        obs2, reward, terminated, truncated, info = env.step(action_vec)

        logps.append(logp)
        rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))
        entropies.append(entropy)

        obs = obs2
        if terminated or truncated:
            break

    # return 계산(누적 감가 보상)
    R, returns = 0.0, []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    returns = torch.cat(returns)                  # (T,)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # baseline 정규화

    logps = torch.cat(logps)                      # (T,)
    entropies = torch.cat(entropies)              # (T,)

    # REINFORCE loss = -(logπ * R) - entropy_bonus
    loss = -(logps * returns.detach()).mean() - entropy_coef * entropies.mean()  # :contentReference[oaicite:2]{index=2}

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(sum(r.item() for r in rewards)), len(rewards)

# ----------- 평가(디터민) -----------
@torch.no_grad()
def run_episode_eval(max_steps=600, thresh=0.5):
    obs, info = env.reset()
    total_r = 0.0
    for t in range(max_steps):
        x = obs_to_tensor(obs)
        spikes, probs = policy.act_inference(x, thresh=thresh)
        action_vec = spikes_to_action(spikes[0].float())
        obs, reward, terminated, truncated, info = env.step(action_vec)
        total_r += reward
        if terminated or truncated:
            break
    return total_r

if __name__ == "__main__":
    # 간단한 학습 루프
    for ep in range(100):
        ret, T = run_episode_train()
        print(f"[train] ep={ep:02d}  return={ret:+.2f}  steps={T}")

    # 평가
    score = run_episode_eval()
    print(f"[eval] return={score:+.2f}")
