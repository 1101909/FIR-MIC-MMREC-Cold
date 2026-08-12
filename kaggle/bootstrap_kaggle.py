"""Standalone Kaggle kernel: clone the public GitHub repository and run it."""
from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/1101909/FIR-MIC-MMREC-Cold.git"
DEST = Path("/kaggle/working/FIR-MIC-MMREC-Cold")
DATA = "/kaggle/input/datasets/toanktx/mmrec-cold"


# Kaggle may currently ship a CUDA 12.8 build that dropped Pascal (sm_60),
# while the assigned Tesla P100 requires it. Install a known compatible wheel
# before importing any project module that imports torch.
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet",
        "--disable-pip-version-check", "--no-cache-dir", "--force-reinstall",
        "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable, "-c",
        "import torch; "
        "print('Torch', torch.__version__, 'architectures', torch.cuda.get_arch_list()); "
        "assert torch.cuda.is_available(); "
        "assert 'sm_60' in torch.cuda.get_arch_list(); "
        "print('GPU', torch.cuda.get_device_name(0), torch.ones(1, device='cuda').item())",
    ],
    check=True,
)

if DEST.exists():
    subprocess.run(["git", "-C", str(DEST), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", REPO, str(DEST)], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", str(DEST / "requirements.txt")], check=True)

subprocess.run(
    [
        sys.executable, "-u", str(DEST / "kaggle/run_gpu.py"),
        "--data-dir", DATA,
        "--datasets", "baby",
        "--seeds", "2022",
        "--epochs", "1",
        "--batch-size", "256",
        "--run-controlled-content-baselines",
        "--run-official-cold-baselines",
    ],
    cwd=DEST,
    check=True,
)
