"""Quick GPU availability check for PyTorch."""

import torch

print("=== GPU Check ===")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name        : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version    : {torch.version.cuda}")
    print(
        f"GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f"Tensor test     : PASSED (shape {y.shape})")
else:
    print("ERROR: No GPU found. Check CUDA install.")
    raise SystemExit(1)
