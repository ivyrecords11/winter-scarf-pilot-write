import os, time, random
import torch
import numpy as np

def _mkdir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_checkpoint(path, policy, optimizer, episode, train_return=None, eval_score=None, extra=None):
    _mkdir(os.path.dirname(path) or ".")
    ckpt = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episode": int(episode),
        "train_return": float(train_return) if train_return is not None else None,
        "eval_score": float(eval_score) if eval_score is not None else None,
        "model_state": policy.state_dict(),
        "optim_state": optimizer.state_dict(),
        # RNG states(재현성)
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        # 필요하면 환경/설정 스냅샷(직렬화 가능한 것만)
        "cfg_repr": repr(globals().get("cfg", None)),
        "extra": extra or {},
    }
    torch.save(ckpt, path)

def load_checkpoint(path, policy, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location or ("cpu" if not torch.cuda.is_available() else None), weights_only=False)
    policy.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt and ckpt["optim_state"] is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    # RNG 복구(선택)
    try:
        rng = ckpt.get("rng", {})
        if rng.get("python") is not None: random.setstate(rng["python"])
        if rng.get("numpy") is not None:  np.random.set_state(rng["numpy"])
        if rng.get("torch_cpu") is not None: torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    except Exception:
        pass  # RNG 복구 실패해도 학습은 계속 가능

    meta = {
        "episode": ckpt.get("episode", 0),
        "train_return": ckpt.get("train_return", None),
        "eval_score": ckpt.get("eval_score", None),
        "timestamp": ckpt.get("timestamp", None),
        "extra": ckpt.get("extra", {}),
    }
    return meta