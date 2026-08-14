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
    parser.add_argument(
        "--fir-mic-max-epochs", type=int,
        help="Maximum validation-selected FIR-MIC epoch; defaults to --epochs",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--semco-batch-size", type=int, default=2048,
        help="SEMCo batch size; 2048 matches the pinned upstream configuration",
    )
    parser.add_argument("--grid-intents", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--grid-dims", nargs="+", type=int, default=[64])
    parser.add_argument("--grid-learning-rates", nargs="+", type=float,
                        default=[5e-4, 1e-3])
    parser.add_argument("--grid-weight-decays", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--grid-dropouts", nargs="+", type=float, default=[0.15])
    parser.add_argument("--grid-router-dropouts", nargs="+", type=float, default=[0.10])
    parser.add_argument("--grid-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-controlled-content-baselines", action="store_true")
    parser.add_argument("--run-official-cold-baselines", action="store_true")
    parser.add_argument("--extended-ablations", action="store_true")
    args = parser.parse_args()
    fir_mic_max_epochs = args.fir_mic_max_epochs or args.epochs
    if args.epochs < 1 or fir_mic_max_epochs < 1:
        parser.error("--epochs and --fir-mic-max-epochs must be positive")
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

    if args.run_official_cold_baselines:
        run(["git", "submodule", "update", "--init", "external/SEMCo", "external/CLCRec"])
        for dataset in args.datasets:
            for seed in args.seeds:
                run([sys.executable,"scripts/run_official_cold_baselines.py","--data-dir",data,"--dataset",dataset,"--models","semco","clcrec","--seed",seed,"--epochs",args.epochs,"--batch-size",args.batch_size,"--semco-batch-size",args.semco_batch_size,"--output",out/"official"/dataset/f"seed_{seed}.json"])

    for dataset in args.datasets:
        for seed in args.seeds:
            run_dir = out / "fir_mic" / dataset / f"seed_{seed}"
            if (args.resume and (run_dir / "result.json").exists()
                    and (run_dir / "optimal_config.json").exists()):
                print("Resume: completed FIR-MIC run exists at", run_dir, flush=True)
                continue
            run([
                sys.executable, "hier_bridge/run_fir_mic_seed.py",
                "--data-dir", data, "--dataset", dataset, "--seed", seed,
                "--epochs", fir_mic_max_epochs, "--batch-size", args.batch_size,
                "--grid-intents", *args.grid_intents,
                "--grid-dims", *args.grid_dims,
                "--grid-learning-rates", *args.grid_learning_rates,
                "--grid-weight-decays", *args.grid_weight_decays,
                "--grid-dropouts", *args.grid_dropouts,
                "--grid-router-dropouts", *args.grid_router_dropouts,
                "--grid-patience", args.grid_patience,
                "--output-dir", run_dir,
                *( ["--resume"] if args.resume else [] ),
                *(["--extended-ablations"] if args.extended_ablations else []),
            ])

    # RQ3 is a post-hoc mechanism analysis over intrinsic content covariates
    # and saved per-user ranks. It does not retrain or select a model.
    for dataset in args.datasets:
        run([
            sys.executable, "scripts/analyze_rq3_semantic_gap.py",
            "--data-dir", data, "--dataset", dataset, "--results-dir", out,
        ])

    run([sys.executable, "scripts/summarize_evidence.py", "--results-dir", out])

    print("Results:", out)


if __name__ == "__main__":
    main()
