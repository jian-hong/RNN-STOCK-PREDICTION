import torch.nn as nn


class StackedGRU(nn.Module):
    """2-layer GRU with BatchNorm (~84K params)."""

    def __init__(self, input_size=1, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.gru1 = nn.GRU(input_size, hidden1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.drop1 = nn.Dropout(dropout)
        self.gru2 = nn.GRU(hidden1, hidden2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.gru1(x)
        out = out[:, -1, :]
        out = self.bn1(out)
        out = self.drop1(out)
        out, _ = self.gru2(out.unsqueeze(1))
        out = out[:, -1, :]
        out = self.bn2(out)
        out = self.drop2(out)
        return self.fc2(self.relu(self.fc1(out)))


if __name__ == "__main__":
    model = StackedGRU()
    print(f"StackedGRU params: {sum(p.numel() for p in model.parameters()):,}")
