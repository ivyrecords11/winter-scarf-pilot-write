# run_rl.py (live viewer 루프를 대체하는 간단 실행/훈련 예시)
import numpy as np
import torch
from torch.distributions import Bernoulli
from mujoco_model import Environment, SimulationConfig
from policy import SpikePolicy
from checkpoint_utils import save_checkpoint, load_checkpoint  # 앞서 만든 유틸

import matplotlib
matplotlib.use("Agg")  # X 없는 환경에서도 저장 가능
import matplotlib.pyplot as plt

train_returns, eval_scores, train_steps = [], [], []

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
    import os
    CKPT_DIR = "checkpoints"
    BEST_PATH = os.path.join(CKPT_DIR, "best.pt")
    RESUME_PATH = os.path.join(CKPT_DIR, "last.pt")   # 최근 저장본에서 재개하고 싶을 때

    # ---- (선택) 재개 ----
    start_ep = 0
    if os.path.exists(RESUME_PATH):
        meta = load_checkpoint(RESUME_PATH, policy, optimizer)
        start_ep = meta["episode"] + 1
        print(f"[resume] {RESUME_PATH} | episode={meta['episode']} eval={meta['eval_score']} time={meta['timestamp']}")

    best_eval = -float("inf")
    if os.path.exists(BEST_PATH):
        # best도 불러오고 싶다면(선택): load_checkpoint(BEST_PATH, policy, optimizer=None)
        best_meta = torch.load(BEST_PATH, map_location="cpu", weights_only=False).get("meta", {})
        best_eval = best_meta.get("eval_score", best_eval)

    EPISODES = 1
    SAVE_EVAL_EVERY = 10

    for ep in range(start_ep, start_ep + EPISODES):
        ret, T = run_episode_train()
        print(f"[train] ep={ep:04d}  return={ret:+.2f}  steps={T}")

        # 주기적 저장
        if (ep + 1) % SAVE_EVAL_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"ep{ep+1}.pt")
            save_checkpoint(path, policy, optimizer, episode=ep, train_return=ret, extra={"steps": T})
            save_checkpoint(RESUME_PATH, policy, optimizer, episode=ep, train_return=ret, extra={"steps": T})
            print(f"[ckpt] saved: {path} & {RESUME_PATH}")

            # 평가 및 best 갱신
            score = run_episode_eval()
            print(f"[eval ] ep={ep:04d}  return={score:+.2f}")

            if score > best_eval:
                best_eval = score
                save_checkpoint(BEST_PATH, policy, optimizer, episode=ep, train_return=ret, eval_score=score)
                print(f"[ckpt] new BEST ({best_eval:+.2f}) -> {BEST_PATH}")

    

    # --- 학습 종료 후: 그래프 저장 ---
    plt.figure()
    plt.plot(train_returns, label="train return")
    plt.plot(eval_scores, label="eval return")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves.png")   # 파일로 저장

    # (선택) CSV로도 남기기
    import csv
    with open("training_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "train_return", "eval_return", "train_steps"])
        for i, (tr, ev, ts) in enumerate(zip(train_returns, eval_scores, train_steps)):
            w.writerow([i + start_ep, tr, ev, ts])