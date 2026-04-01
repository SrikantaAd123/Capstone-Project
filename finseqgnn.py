import torch
import torch.nn as nn

class FinSeqGNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.price_lstm = nn.LSTM(5, 64, batch_first=True)
        self.sent_lstm  = nn.LSTM(1, 32, batch_first=True)

        self.fc = nn.Linear(96, 1)

    def forward(self, xp, xs):

        _, (hp, _) = self.price_lstm(xp)

        xs = xs.unsqueeze(-1)
        _, (hs, _) = self.sent_lstm(xs)

        out = torch.cat([hp[-1], hs[-1]], dim=1)

        return self.fc(out).squeeze()
