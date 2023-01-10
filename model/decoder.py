import torch
from torch import nn

from utils.arguments import arguments


class SimpleDecoder(nn.Module):
    def __init__(self, in_len: int, out_len: int, arg, rnn='LSTM'):
        super().__init__()

        hidden_size = 512
        self.fnn = nn.Sequential(
            nn.Linear(hidden_size, out_len),
            nn.Softmax(dim=1)
        )
        self.rnn = getattr(nn, rnn)(input_size=in_len, hidden_size=hidden_size // 2, num_layers=arg.num_layer,
                          batch_first=True, bidirectional=True, dropout=0.1)

    def forward(self, x):
        x = self.rnn(x)[0]
        return self.fnn(x)


class MultiTurnDecoder(nn.Module):
    def __init__(self, in_len: int, out_len: int, arg, encoder='GRU', cencoder='GRU'):
        super().__init__()

        self._memory = []

        # the length of knowledge vectors must be equal to that of input vectors
        self._knowledge_len = in_len
        self._memory_len = 256

        self.encoder = getattr(nn, encoder)(input_size=in_len, hidden_size=self._memory_len // 2, num_layers=1,
                              batch_first=True, bidirectional=True, dropout=0.1)
        self.contextual_encoder = getattr(nn, cencoder)(input_size=in_len, hidden_size=self._memory_len // 2, num_layers=1,
                                         batch_first=True, bidirectional=True, dropout=0.1)
        self.knowledge_encoder = nn.Linear(self._memory_len, self._knowledge_len)
        self.decoder = SimpleDecoder(in_len, out_len)

    def forward(self, c):
        c_memory = self.contextual_encoder(c)[0][-1, :]
        u = self.encoder(c)[0][-1, :]
        p = torch.zeros(len(self._memory))
        if len(self._memory) > 0:
            for i in range(len(self._memory)):
                p[i] = torch.inner(self._memory[i], u)
            p = torch.softmax(p, dim=0)
            h = sum(self._memory[i] * p[i] for i in range(len(self._memory)))
        else:
            h = torch.zeros(self._memory_len)
        h = h.to(arguments.device)
        o = self.knowledge_encoder(u + h).to(arguments.device)
        self._memory.append(c_memory.detach())
        return self.decoder(c + o)

    def reset(self) -> None:
        self._memory.clear()
