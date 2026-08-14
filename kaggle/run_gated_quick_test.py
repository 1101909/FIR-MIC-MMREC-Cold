"""Rebuild one final FIR-MIC model from GitHub-locked logs, then test gating.

This deliberately skips hyperparameter search, baselines and ablations.  The
only fit is the confirmatory FIR-MIC model for the already selected number of
epochs.  It exists because historical GitHub artifacts retained experiment
logs/configuration but not learned weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HIER = ROOT / "hier_bridge"
sys.path.insert(0, str(HIER))
sys.path.insert(0, str(ROOT))

from run_fir_mic_seed import FIRMIC, reset_random_state, teacher_for_modality, train_model
from run_intent_interpolation_seed import l2_blocks
from run_mmrec_seq_scl import Config, build_samples, leakage_checks, read_interactions, temporal_item_split
from run_soft_intent_bridge_v2_seed import pseudo_cold


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_json(ref: str, path: str) -> dict:
    try:
        raw = subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"{ref}:{path}"],
            cwd=ROOT, text=True, encoding="utf8",
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Cannot read {path} from {ref}. Fetch the GitHub results branch first."
        ) from error
    return json.loads(raw)


def compact_state_dict(model: torch.nn.Module) -> dict:
    """Keep learned weights only; feature/teacher buffers come from MMREC data."""
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name not in {"image", "text", "teacher"}
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick Gated-FIR-MIC test using locked GitHub experiment logs"
    )
    parser.add_argument("--dataset", choices=("baby", "clothing", "sports"), default="baby")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-branch", default="kaggle-results-gridsearch-v1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main():
    args = parse_args()
    output_dir = args.output_dir or (
        ROOT / "results" / "gated_rebuild" / args.dataset / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_fetch:
        print(f"Fetching locked logs from GitHub branch {args.results_branch}...", flush=True)
        subprocess.run(
            [
                "git", "-c", f"safe.directory={ROOT.as_posix()}", "fetch", "origin",
                f"+refs/heads/{args.results_branch}:refs/remotes/origin/{args.results_branch}",
            ],
            cwd=ROOT, check=True,
        )
    results_ref = f"origin/{args.results_branch}"
    artifact_root = f"results/fir_mic/{args.dataset}/seed_{args.seed}"
    optimal = git_json(results_ref, f"{artifact_root}/optimal_config.json")
    locked = optimal.get("locked_score_weights")
    required = {
        "intents", "dim", "learning_rate", "weight_decay", "dropout",
        "router_dropout", "best_epoch",
    }
    missing = sorted(required - set(optimal))
    if missing or not locked:
        raise RuntimeError(f"Incomplete GitHub optimal_config; missing={missing}, locked={locked}")

    print(json.dumps({
        "mode": "rebuild-confirmatory-only",
        "dataset": args.dataset,
        "seed": args.seed,
        "github_ref": results_ref,
        "skipped": ["grid search", "baselines", "modality ablations"],
        "fit_epochs": optimal["best_epoch"],
        "optimal": {key: optimal[key] for key in sorted(required)},
        "locked_score_weights": locked,
    }, indent=2), flush=True)

    data_root = args.data_dir / args.dataset
    interactions = read_interactions(data_root, args.dataset)
    warm, validation_cold, test_cold, cutoff, _ = temporal_item_split(
        interactions, Config(max_sequence_length=10)
    )
    train, validation, test, _ = build_samples(
        interactions, warm, validation_cold, test_cold, cutoff, 10
    )
    checks = leakage_checks(train, validation, test, warm, validation_cold, test_cold)
    pseudo_train = pseudo_cold(train)

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU must be enabled for the quick rebuild")
    device = torch.device("cuda")
    image_np = l2_blocks(np.load(data_root / "image_feat.npy", mmap_mode="r"))
    text_np = l2_blocks(np.load(data_root / "text_feat.npy", mmap_mode="r"))
    image = torch.from_numpy(image_np).to(device)
    text = torch.from_numpy(text_np).to(device)

    reset_random_state(args.seed)
    teacher, intent_diag = teacher_for_modality(
        image_np, text_np, warm, optimal["intents"], args.seed, "full"
    )
    model = FIRMIC(
        image, text, torch.from_numpy(teacher).to(device), optimal["intents"],
        optimal["dim"], 10, modality="full", dropout=optimal["dropout"],
        router_dropout=optimal["router_dropout"],
    ).to(device)
    started = time.perf_counter()
    training = train_model(
        model, pseudo_train, image, text, optimal["best_epoch"],
        args.batch_size, args.seed, device,
        learning_rate=optimal["learning_rate"], weight_decay=optimal["weight_decay"],
    )
    rebuild_seconds = time.perf_counter() - started
    write_csv(output_dir / "training_rebuild.csv", training)
    (output_dir / "optimal_config.json").write_text(
        json.dumps(optimal, indent=2), encoding="utf8"
    )
    (output_dir / "locked_config.json").write_text(
        json.dumps(locked, indent=2), encoding="utf8"
    )
    audit = {
        "mode": "confirmatory-only reconstruction from GitHub-locked logs",
        "github_results_ref": results_ref,
        "github_artifact_root": artifact_root,
        "no_grid_search": True,
        "no_baselines": True,
        "no_ablations": True,
        "fit_epochs": optimal["best_epoch"],
        "rebuild_seconds": rebuild_seconds,
        "train_samples": len(train),
        "pseudo_cold_train_samples": len(pseudo_train),
        "validation_users": len(validation),
        "test_users": len(test),
        "leakage_checks": checks,
        "intent_initialization": intent_diag,
    }
    (output_dir / "rebuild_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf8"
    )
    torch.save({
        "state_dict": compact_state_dict(model),
        "optimal_config": optimal,
        "rebuild_audit": audit,
    }, output_dir / "model.pt")
    print(
        f"Saved compact learned checkpoint ({(output_dir / 'model.pt').stat().st_size / 2**20:.1f} MiB)",
        flush=True,
    )

    del model, image, text, teacher
    torch.cuda.empty_cache()
    gated_output = output_dir / "gated_posthoc"
    command = [
        sys.executable, "-u", str(HIER / "run_gated_fir_mic_posthoc.py"),
        "--data-dir", str(args.data_dir), "--dataset", args.dataset,
        "--seed", str(args.seed), "--batch-size", str(args.batch_size),
        "--run-dir", str(output_dir), "--output-dir", str(gated_output),
        "--device", "cuda",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    result = json.loads((gated_output / "gated_test_metrics.json").read_text(encoding="utf8"))
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "fit_epochs": optimal["best_epoch"],
        "original": result["original"],
        "gated": result["gated"],
        "relative_gain": result["relative_gain"],
        "selected_gate": result["selected_gate"],
        "q4_original": next(
            row for row in result["by_semantic_gap"]
            if row["variant"] == "FIR-MIC original" and row["gap_group"] == "Q4-high"
        ),
        "q4_gated": next(
            row for row in result["by_semantic_gap"]
            if row["variant"] == "Gated-FIR-MIC" and row["gap_group"] == "Q4-high"
        ),
    }
    (output_dir / "quick_test_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf8"
    )
    print("QUICK TEST SUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
