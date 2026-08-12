"""Kaggle GPU entry point for MMREC-COLD.

Clone the repository in a Kaggle cell, then run this file.  It never downloads
or silently substitutes a baseline implementation.  Eligible upstream models
must be present at their pinned submodule commits.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def resolve_mmrec(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += [
        Path("/kaggle/input/mmrec-cold"),
        Path("/kaggle/input/datasets/toanktx/mmrec-cold"),
        Path("/kaggle/input/mmreccold"),
    ]
    for candidate in candidates:
        if all((candidate / name / f"{name}.inter").exists() for name in ("baby", "clothing", "sports")):
            return candidate
    raise FileNotFoundError("MMREC-COLD not found. Attach Kaggle dataset toanktx/mmrec-cold.")


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--datasets", nargs="+", default=["baby"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2022])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--run-controlled-content-baselines", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not enabled")
    print("GPU", torch.cuda.get_device_name(0), "Torch", torch.__version__)
    data = resolve_mmrec(args.data_dir)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    audit = json.loads((ROOT / "github_baselines_manifest.json").read_text())
    (out / "baseline_source_audit.json").write_text(json.dumps(audit, indent=2))

    # These are controlled content baselines in this repository, not claimed as
    # unchanged official GitHub reproductions. They share the exact evaluator.
    if args.run_controlled_content_baselines:
        run([
            sys.executable, "run_mmrec_seq_scl.py", "--data-dir", data,
            "--datasets", *args.datasets, "--models", "static", "multi-item", "multi-cluster",
            "--seeds", *args.seeds, "--epochs", args.epochs, "--device", "cuda",
            "--output-dir", out / "controlled_content", "--resume",
        ])

    for dataset in args.datasets:
        for seed in args.seeds:
            run([
                sys.executable, "hier_bridge/run_fir_mic_seed.py",
                "--data-dir", data, "--dataset", dataset, "--seed", seed,
                "--epochs", args.epochs, "--batch-size", args.batch_size,
                "--output-dir", out / "fir_mic" / dataset / f"seed_{seed}",
            ])

    print("Results:", out)


if __name__ == "__main__":
    main()
