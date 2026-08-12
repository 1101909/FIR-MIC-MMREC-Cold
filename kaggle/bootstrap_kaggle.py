"""Standalone Kaggle kernel: clone the public GitHub repository and run it."""
from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/1101909/FIR-MIC-MMREC-Cold.git"
DEST = Path("/kaggle/working/FIR-MIC-MMREC-Cold")
DATA = "/kaggle/input/datasets/toanktx/mmrec-cold"

if DEST.exists():
    subprocess.run(["git", "-C", str(DEST), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO, str(DEST)], check=True)

subprocess.run(
    [
        sys.executable, "-u", str(DEST / "kaggle/run_gpu.py"),
        "--data-dir", DATA,
        "--datasets", "baby",
        "--seeds", "2022",
        "--epochs", "1",
        "--batch-size", "256",
        "--run-controlled-content-baselines",
    ],
    cwd=DEST,
    check=True,
)
