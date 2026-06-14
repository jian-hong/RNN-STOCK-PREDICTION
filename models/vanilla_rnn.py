import torch.nn as nn


class VanillaRNN(nn.Module):
    """2-layer SimpleRNN baseline (~8K params)."""

    def __init__(
        self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, window=60
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        return self.fc2(out)


if __name__ == "__main__":
    model = VanillaRNN()
    print(f"VanillaRNN params: {sum(p.numel() for p in model.parameters()):,}")
