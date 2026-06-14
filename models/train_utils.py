"""
Shared device, training loop, and evaluation helpers for all PyTorch models.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockSequenceDataset(Dataset):
    """Wraps numpy X, y arrays into a PyTorch Dataset."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    train_ds = StockSequenceDataset(X_train, y_train)
    val_ds = StockSequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs=100,
    patience=15,
    scheduler=None,
    verbose=True,
):
    """Generic training loop with EarlyStopping."""
    model.to(DEVICE)
    best_val_loss = float("inf")
    patience_count = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch).squeeze(-1)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler:
            scheduler.step(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch+1:3d} | train={train_loss:.5f} | val={val_loss:.5f} | lr={lr:.2e}"
            )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                if verbose:
                    print(f"  EarlyStopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return history


def predict_array(model, X, batch_size=64):
    """Run inference on numpy X, return numpy predictions."""
    model.eval()
    model.to(DEVICE)
    ds = StockSequenceDataset(X, np.zeros(len(X)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            out = model(X_batch).squeeze(-1)
            preds.append(out.cpu().numpy())
    return np.concatenate(preds)


def predict_quantile_array(model, X, batch_size=64):
    """Run inference on QuantileLSTM, return q05/q50/q95 numpy arrays."""
    model.eval()
    model.to(DEVICE)
    ds = StockSequenceDataset(X, np.zeros(len(X)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    q05_list, q50_list, q95_list = [], [], []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            q05, q50, q95 = model(X_batch)
            q05_list.append(q05.squeeze(-1).cpu().numpy())
            q50_list.append(q50.squeeze(-1).cpu().numpy())
            q95_list.append(q95.squeeze(-1).cpu().numpy())
    return (
        np.concatenate(q05_list),
        np.concatenate(q50_list),
        np.concatenate(q95_list),
    )


def load_model_weights(model_class, path):
    """Instantiate model class and load state dict from .pt checkpoint."""
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model
