"""
run_phase2.py — orchestrates Phase 2 pipeline (PyTorch + WSL2).

Usage:
  python scripts/run_phase2.py --step all
  python scripts/run_phase2.py --step check
  python scripts/run_phase2.py --step slim
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
PYTHON = sys.executable

STEPS = {
    "check": [PYTHON, str(SCRIPTS / "check_gpu.py")],
    "slim": [PYTHON, str(SCRIPTS / "03_train.py"), "--all-slim"],
    "quantile": [PYTHON, str(SCRIPTS / "03b_train_quantile.py")],
    "eval_slim": [PYTHON, str(SCRIPTS / "04e_evaluate_slim.py")],
    "eval_quantile": [PYTHON, str(SCRIPTS / "04c_evaluate_quantile.py")],
    "mc": [PYTHON, str(SCRIPTS / "04b_mc_evaluate.py"), "--all"],
    "evaluate": [PYTHON, str(SCRIPTS / "04_evaluate.py"), "--all"],
    "eval_arima_ci": [PYTHON, str(SCRIPTS / "04d_evaluate_arima_ci.py")],
    "walkfwd": [PYTHON, str(SCRIPTS / "06_walkforward.py")],
    "visualise": [PYTHON, str(SCRIPTS / "05_visualise.py"), "--phase2"],
    "summary": [PYTHON, str(SCRIPTS / "07_summary.py")],
}

DEFAULT_ALL_ORDER = [
    "check",
    "slim",
    "quantile",
    "eval_slim",
    "eval_quantile",
    "mc",
    "eval_arima_ci",
    "walkfwd",
    "visualise",
    "summary",
]


def run_step(name, cmd, walkforward_all=False):
    if name == "walkfwd" and walkforward_all:
        cmd = [PYTHON, str(SCRIPTS / "06_walkforward.py"), "--all"]
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"ERROR in step '{name}' — stopping.")
        sys.exit(result.returncode)
    print(f"Step '{name}' completed.")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 pipeline (PyTorch)")
    parser.add_argument(
        "--step",
        default="all",
        choices=list(STEPS.keys()) + ["all"],
        help="Phase 2 step to run",
    )
    parser.add_argument(
        "--walkforward-all",
        action="store_true",
        help="When running walkfwd step, use all 10 tickers",
    )
    args = parser.parse_args()

    if args.step == "all":
        for step_name in DEFAULT_ALL_ORDER:
            run_step(step_name, STEPS[step_name], walkforward_all=args.walkforward_all)
    else:
        run_step(args.step, STEPS[args.step], walkforward_all=args.walkforward_all)

    print("\nAll steps complete.")


if __name__ == "__main__":
    main()
