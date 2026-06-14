"""
run_pipeline.py
Orchestrates the full modular pipeline inside the project venv.
Usage: python scripts/run_pipeline.py [--step download|preprocess|train|evaluate|visualise|verify|all]
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"
PYTHON = sys.executable

STEPS = ["download", "preprocess", "train", "evaluate", "visualise", "verify"]


def run_step(step):
    commands = {
        "download": [PYTHON, str(SCRIPTS / "01_download_data.py")],
        "preprocess": [PYTHON, str(SCRIPTS / "02_preprocess.py")],
        "train": [PYTHON, str(SCRIPTS / "03_train.py"), "--all"],
        "evaluate": [PYTHON, str(SCRIPTS / "04_evaluate.py"), "--all"],
        "visualise": [PYTHON, str(SCRIPTS / "05_visualise.py")],
        "verify": [PYTHON, str(SCRIPTS / "00_verify.py")],
    }
    cmd = commands[step]
    print(f"\n{'='*60}\nSTEP: {step.upper()}\n{'='*60}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"ERROR: Step '{step}' failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Step '{step}' completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run RNN stock prediction pipeline")
    parser.add_argument(
        "--step",
        choices=STEPS + ["all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    args = parser.parse_args()

    if args.step == "all":
        for step in STEPS:
            run_step(step)
    else:
        run_step(args.step)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
