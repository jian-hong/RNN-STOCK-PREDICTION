import torch.nn as nn


class StackedLSTM(nn.Module):
    """2-layer LSTM with BatchNorm (~112K params)."""

    def __init__(self, input_size=1, hidden1=128, hidden2=64, dropout=0.2, window=60):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = out[:, -1, :]
        out = self.bn1(out)
        out = self.drop1(out)
        out, _ = self.lstm2(out.unsqueeze(1))
        out = out[:, -1, :]
        out = self.bn2(out)
        out = self.drop2(out)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return self.fc3(out)


if __name__ == "__main__":
    model = StackedLSTM()
    print(f"StackedLSTM params: {sum(p.numel() for p in model.parameters()):,}")
