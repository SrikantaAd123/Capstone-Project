import torch
import torch.nn as nn

class FINseqGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_price = nn.LSTM(5, 64, batch_first=True)
        self.lstm_sent = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(96, 1)

    def forward(self, xp, xs):
        _, (hp, _) = self.lstm_price(xp)
        xs = xs.unsqueeze(-1)
        _, (hs, _) = self.lstm_sent(xs)
        out = torch.cat([hp[-1], hs[-1]], dim=1)
        return self.fc(out).squeeze()
