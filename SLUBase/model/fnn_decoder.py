import torch
import torch.nn as nn


class FNNDecoder(nn.Module):
    def __init__(self, in_len: int, out_len: int):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(in_len, out_len),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fnn(x)
