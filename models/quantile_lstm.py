"""
QuantileLSTM — shared encoder + q05/q50/q95 heads via pinball loss.
"""

import torch
import torch.nn as nn


def pinball_loss(quantile):
    """Asymmetric quantile loss. q=0.5 → MAE."""

    def loss_fn(y_pred, y_true):
        error = y_true - y_pred
        return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

    return loss_fn


class QuantileLSTM(nn.Module):
    def __init__(self, input_size=1, hidden1=64, hidden2=32, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden1)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, num_layers=1, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden2)
        self.drop2 = nn.Dropout(dropout)

        self.head_q05 = nn.Sequential(
            nn.Linear(hidden2, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.head_q50 = nn.Sequential(
            nn.Linear(hidden2, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.head_q95 = nn.Sequential(
            nn.Linear(hidden2, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.ln2(out)
        out = self.drop2(out)
        return self.head_q05(out), self.head_q50(out), self.head_q95(out)


if __name__ == "__main__":
    model = QuantileLSTM()
    print(f"QuantileLSTM params: {sum(p.numel() for p in model.parameters()):,}")
