# GPU Setup Guide — Phase 2 (PyTorch)

## Why PyTorch

TensorFlow max CUDA is 12.x. Machines with CUDA 13.0 drivers (e.g. RTX 4050/5060) work best with **PyTorch cu121 wheels** (backward-compatible on newer drivers) or **nightly cu130**.

## Recommended: WSL2 + PyTorch cu121

```powershell
# Windows PowerShell (admin)
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

```bash
# Inside WSL2 — use native Linux FS (not /mnt/c/...) for speed
mkdir -p ~/projects
cp -r /mnt/c/Users/<you>/FSA1/RNN-STOCK-PREDICTION ~/projects/rnn-stock-prediction
cd ~/projects/rnn-stock-prediction

python3 --version          # confirm 3.12.x
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch CUDA 12.1 wheels (runs on CUDA 13.0 driver via backward compat)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# If torch.cuda.is_available() is False, try nightly cu130:
# pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

pip install -r requirements.txt

python scripts/check_gpu.py
# Expected: True / NVIDIA GeForce RTX 4050 Laptop GPU
```

### Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Expected speedup (RTX 4050 vs CPU)

| Step | CPU | GPU (est.) |
|------|-----|------------|
| SlimLSTM × 10 | ~15 min | ~3–5 min |
| QuantileLSTM × 10 | ~15 min | ~3–5 min |
| MC Dropout (200×20) | ~8+ hrs | ~20–30 min |
| ARIMA walk-forward | ~11 min | ~11 min (CPU only) |
| Walk-forward × 3 | ~30 min | ~10–15 min |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available() = False` with cu121 | Try nightly cu130 wheels |
| `nvidia-smi` works but PyTorch can't see GPU | Confirm you're in WSL2, not Windows PowerShell |
| Slow pip install | Confirm `pwd` shows `/home/...` not `/mnt/c/...` |
| OOM on RTX 4050 (4–6 GB VRAM) | Reduce `batch_size` from 32 to 16 in `models/train_utils.py` |

---

## CPU fallback

All scripts run on CPU if no GPU is detected (`DEVICE` auto-selects in `train_utils.py`). MC Dropout is the slowest step on CPU.
