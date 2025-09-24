# policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikePolicy(nn.Module):
    """
    input: 10x10 boolean (float32로 바꿔서 사용)
    output: 4 logits ( +x, -x, +y, -y )
    """
    def __init__(self, hidden=64, p_dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),               # 100
            nn.Linear(100, hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden, 4)        # logits (unnormalized)
        )

    def forward(self, x_bool_bchw):
        """
        x_bool_bchw: shape (B, 1, 10, 10), dtype float32 in {0.,1.}
        return: logits shape (B, 4)
        """
        return self.net(x_bool_bchw)

    @torch.no_grad()
    def act_inference(self, x_bool_bchw, thresh=0.5):
        """
        평가/실행 시: 시그모이드 확률>thresh → {0,1} 스파이크
        """
        logits = self.forward(x_bool_bchw)                 # (B,4)
        probs  = torch.sigmoid(logits)                     # (B,4) in [0,1]
        spikes = (probs >= thresh).to(torch.uint8)         # bool spike
        return spikes, probs
