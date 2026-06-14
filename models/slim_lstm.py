"""
SlimLSTM — LayerNorm instead of BatchNorm (~28K params).
Robust to distribution shift across market regimes.
"""

import torch.nn as nn


class SlimLSTM(nn.Module):
    def __init__(self, input_size=1, hidden1=64, hidden2=32, dropout=0.15):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden1)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, num_layers=1, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden2)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.ln1(out)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.ln2(out)
        out = self.drop2(out)
        return self.fc2(self.relu(self.fc1(out)))


if __name__ == "__main__":
    model = SlimLSTM()
    print(f"SlimLSTM params: {sum(p.numel() for p in model.parameters()):,}")
