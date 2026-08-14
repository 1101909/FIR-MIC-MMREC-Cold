"""Post-hoc confidence-aware gating for an already trained FIR-MIC model.

This runner never calls an optimizer and never changes model parameters.  It
loads ``model.pt``, rebuilds the validation/test score components, selects a
small gate grid on validation only, and evaluates the locked gate on test.

    alpha(u,c) = sigmoid(w1 * s_anchor(u,c)
                         + w2 * percentile_anchor(u,c) + bias)
    score(u,c) = alpha * z(s_anchor) + (1-alpha) * auxiliary

where ``auxiliary`` is the original validation-locked weighted sum of the
routed, future and fine-residual branches.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_fir_mic_seed import FIRMIC, make_cache, metrics_from_ranks, stable_rank, z
from run_intent_interpolation_seed import l2_blocks
from run_mmrec_seq_scl import Config, build_samples, leakage_checks, read_interactions, temporal_item_split
from analyze_rq3_semantic_gap import semantic_gap_rows


METRICS = ("Recall@10", "NDCG@10", "MRR@10", "Recall@20", "NDCG@20", "MRR@20")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def anchor_percentile(values: np.ndarray) -> np.ndarray:
    """Empirical percentile in [0,1], with identical scores sharing a rank."""
    values = np.asarray(values)
    if len(values) <= 1:
        return np.ones_like(values, dtype=np.float32)
    ordered = np.sort(values)
    upper_rank = np.searchsorted(ordered, values, side="right") - 1
    return (upper_rank / (len(values) - 1)).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def prepare_rows(cache: list[dict], locked: dict) -> list[dict]:
    prepared = []
    for row in cache:
        anchor = np.asarray(row["anchor"])
        prepared.append({
            "sample": row["sample"],
            "target_pos": row["target_pos"],
            "anchor": anchor,
            "anchor_z": z(anchor),
            "anchor_percentile": anchor_percentile(anchor),
            "auxiliary": (
                locked["lambda_z"] * z(row["routed"])
                + locked["lambda_next"] * z(row["next"])
                + locked["lambda_fine"] * z(row["fine_residual"])
            ),
        })
    return prepared


def original_ranks(rows: list[dict]) -> list[int]:
    return [stable_rank(row["anchor_z"] + row["auxiliary"], row["target_pos"]) for row in rows]


def gated_ranks(rows: list[dict], w1: float, w2: float, bias: float):
    ranks, target_alphas = [], []
    for row in rows:
        alpha = sigmoid(w1 * row["anchor"] + w2 * row["anchor_percentile"] + bias)
        score = alpha * row["anchor_z"] + (1.0 - alpha) * row["auxiliary"]
        ranks.append(stable_rank(score, row["target_pos"]))
        target_alphas.append(float(alpha[row["target_pos"]]))
    return ranks, target_alphas


def subset_metrics(ranks: list[int], indices: list[int]) -> dict:
    return metrics_from_ranks([ranks[index] for index in indices])


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def validation_hard_split(rows: list[dict]):
    """Bottom quartile of target anchor percentile; no gap/test labels used."""
    ordered = sorted(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["anchor_percentile"][rows[index]["target_pos"]]), index
        ),
    )
    hard_count = max(1, math.ceil(len(rows) / 4))
    hard = sorted(ordered[:hard_count])
    hard_set = set(hard)
    easy = [index for index in range(len(rows)) if index not in hard_set]
    return hard, easy


def select_gate(rows: list[dict], args):
    hard, easy = validation_hard_split(rows)
    baseline_ranks = original_ranks(rows)
    baseline_all = metrics_from_ranks(baseline_ranks)
    baseline_hard = subset_metrics(baseline_ranks, hard)
    baseline_easy = subset_metrics(baseline_ranks, easy)
    baseline_row = {
        "mode": "original", "w1": "", "w2": "", "bias": "", "eligible": 1,
        "mean_target_alpha": "",
        **prefixed("overall", baseline_all),
        **prefixed("hard_proxy", baseline_hard),
        **prefixed("easy_proxy", baseline_easy),
    }

    grid_rows, eligible_rows = [baseline_row], []
    for w1, w2, bias in itertools.product(args.gate_w1, args.gate_w2, args.gate_bias):
        ranks, target_alphas = gated_ranks(rows, w1, w2, bias)
        overall = metrics_from_ranks(ranks)
        hard_metrics = subset_metrics(ranks, hard)
        easy_metrics = subset_metrics(ranks, easy)
        eligible = (
            overall["NDCG@10"] >= baseline_all["NDCG@10"] - args.max_overall_ndcg_drop
            and easy_metrics["NDCG@10"] >= baseline_easy["NDCG@10"] - args.max_easy_ndcg_drop
        )
        row = {
            "mode": "gated", "w1": w1, "w2": w2, "bias": bias,
            "eligible": int(eligible), "mean_target_alpha": float(np.mean(target_alphas)),
            **prefixed("overall", overall),
            **prefixed("hard_proxy", hard_metrics),
            **prefixed("easy_proxy", easy_metrics),
        }
        grid_rows.append(row)
        if eligible:
            eligible_rows.append(row)

    candidates = eligible_rows or grid_rows[1:]
    best = max(candidates, key=lambda row: (
        row["hard_proxy_NDCG@10"], row["hard_proxy_Recall@10"],
        row["hard_proxy_MRR@10"], row["hard_proxy_NDCG@20"],
        row["overall_NDCG@10"], row["overall_Recall@10"],
    ))
    return grid_rows, best, {
        "overall": baseline_all, "hard_proxy": baseline_hard,
        "easy_proxy": baseline_easy, "hard_users": len(hard), "easy_users": len(easy),
    }


def load_checkpoint_model(checkpoint_path: Path, optimal: dict, image, text, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint has no state_dict: {checkpoint_path}")

    expected_shapes = {"image": tuple(image.shape), "text": tuple(text.shape)}
    for name, expected in expected_shapes.items():
        if name in state and tuple(state[name].shape) != expected:
            raise RuntimeError(
                f"Checkpoint {name} shape {tuple(state[name].shape)} does not match data {expected}"
            )

    # These buffers are data, not learned weights.  Use the current normalized
    # feature arrays and an unused teacher placeholder during inference.
    data_buffers = {"image", "text", "teacher"}
    for name in data_buffers:
        state.pop(name, None)
    teacher = torch.zeros((image.shape[0], optimal["intents"]), device=device)
    model = FIRMIC(
        image, text, teacher, optimal["intents"], optimal["dim"], 10,
        modality="full", dropout=optimal["dropout"],
        router_dropout=optimal["router_dropout"],
    ).to(device)
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != data_buffers or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Incompatible checkpoint; missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model


def metrics_by_gap(ranks: list[int], gap_rows: list[dict], variant: str) -> list[dict]:
    output = []
    for group in ("Q1-low", "Q2", "Q3", "Q4-high"):
        indices = [index for index, row in enumerate(gap_rows) if row["gap_group"] == group]
        output.append({"variant": variant, "gap_group": group, "users": len(indices),
                       **subset_metrics(ranks, indices)})
    return output


def relative_gain(new: dict, old: dict) -> dict:
    return {
        key: (new[key] / old[key] - 1.0) if old[key] else None
        for key in METRICS
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-only confidence-aware anchor gating for FIR-MIC"
    )
    parser.add_argument("--dataset", choices=("baby", "clothing", "sports"), default="baby")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--data-dir", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path,
                        help="Existing seed directory containing model.pt and optimal_config.json")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to local results/fir_mic/DATASET/seed_SEED/gated_posthoc")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gate-w1", nargs="+", type=float, default=[0.0, 2.0, 4.0, 8.0])
    parser.add_argument("--gate-w2", nargs="+", type=float, default=[0.0, 2.0, 4.0, 8.0])
    parser.add_argument("--gate-bias", nargs="+", type=float, default=[-8.0, -6.0, -4.0, -2.0, 0.0])
    parser.add_argument("--max-overall-ndcg-drop", type=float, default=0.001)
    parser.add_argument("--max-easy-ndcg-drop", type=float, default=0.001)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if any(value < 0 for value in args.gate_w1 + args.gate_w2):
        parser.error("--gate-w1 and --gate-w2 must be non-negative confidence slopes")
    if args.max_overall_ndcg_drop < 0 or args.max_easy_ndcg_drop < 0:
        parser.error("NDCG drop tolerances must be non-negative")
    return args


def discover_run_dir(dataset: str, seed: int, local_run_dir: Path) -> Path:
    """Find a preserved Kaggle Notebook Output when the working copy lacks it."""
    if (local_run_dir / "model.pt").exists():
        return local_run_dir
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return local_run_dir
    suffix = Path("results") / "fir_mic" / dataset / f"seed_{seed}" / "model.pt"
    matches = sorted(
        path for path in kaggle_input.glob(f"**/{suffix.as_posix()}")
        if path.is_file()
    )
    if not matches:
        return local_run_dir
    if len(matches) > 1:
        print(
            "Multiple attached checkpoints found; using the first one: "
            f"{matches[0]}. Override with --run-dir if needed.",
            flush=True,
        )
    else:
        print(f"Using attached Kaggle checkpoint: {matches[0]}", flush=True)
    return matches[0].parent


def main():
    args = parse_args()
    local_run_dir = ROOT / "results" / "fir_mic" / args.dataset / f"seed_{args.seed}"
    run_dir = args.run_dir or discover_run_dir(args.dataset, args.seed, local_run_dir)
    # Kaggle inputs are read-only, so recovered checkpoints must still write
    # their small post-hoc artifacts to the working repository.
    output_dir = args.output_dir or local_run_dir / "gated_posthoc"
    checkpoint_path = run_dir / "model.pt"
    optimal_path = run_dir / "optimal_config.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Post-hoc gating needs the trained model.pt; "
            "the uploaded rank/result CSV files cannot reconstruct candidate-level component scores. "
            "Attach the previous Kaggle Notebook Output that contains this checkpoint "
            "as an input, or run in the preserved session that still contains model.pt."
        )
    if not optimal_path.exists():
        raise FileNotFoundError(f"Missing {optimal_path}")
    optimal = json.loads(optimal_path.read_text(encoding="utf8"))
    locked = optimal.get("locked_score_weights")
    if not locked:
        locked_path = run_dir / "locked_config.json"
        if not locked_path.exists():
            raise RuntimeError("No validation-locked FIR-MIC score weights were found")
        locked = json.loads(locked_path.read_text(encoding="utf8"))

    data_root = args.data_dir / args.dataset
    interactions = read_interactions(data_root, args.dataset)
    warm, validation_cold, test_cold, cutoff, _ = temporal_item_split(
        interactions, Config(max_sequence_length=10)
    )
    train, validation, test, _ = build_samples(
        interactions, warm, validation_cold, test_cold, cutoff, 10
    )
    leakage_checks(train, validation, test, warm, validation_cold, test_cold)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    image_np = l2_blocks(np.load(data_root / "image_feat.npy", mmap_mode="r"))
    text_np = l2_blocks(np.load(data_root / "text_feat.npy", mmap_mode="r"))
    image = torch.from_numpy(image_np).to(device)
    text = torch.from_numpy(text_np).to(device)
    model = load_checkpoint_model(checkpoint_path, optimal, image, text, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Recomputing validation components (inference only)...", flush=True)
    validation_cache = make_cache(
        model, validation, sorted(validation_cold), image, text, args.batch_size, device
    )
    validation_rows = prepare_rows(validation_cache, locked)
    del validation_cache
    grid, selected, baseline_validation = select_gate(validation_rows, args)
    write_csv(output_dir / "gated_validation_grid.csv", grid)
    gate = {key: float(selected[key]) for key in ("w1", "w2", "bias")}
    locked_output = {
        "method": "Gated-FIR-MIC post-hoc",
        "formula": "alpha*z(anchor) + (1-alpha)*auxiliary",
        "alpha": "sigmoid(w1*raw_anchor + w2*candidate_anchor_percentile + bias)",
        "original_score_weights": locked,
        "selected_gate": gate,
        "selected_gate_satisfied_guardrails": bool(selected["eligible"]),
        "selection": "maximize hard-proxy validation NDCG@10; test labels never used",
        "hard_proxy": "bottom quartile of validation target anchor percentile",
        "guardrails": {
            "max_overall_validation_ndcg10_drop": args.max_overall_ndcg_drop,
            "max_easy_validation_ndcg10_drop": args.max_easy_ndcg_drop,
        },
        "baseline_validation": baseline_validation,
        "no_retraining": True,
    }
    (output_dir / "gated_locked_config.json").write_text(
        json.dumps(locked_output, indent=2), encoding="utf8"
    )
    del validation_rows

    print(f"Locked gate: {gate}. Recomputing test components (inference only)...", flush=True)
    test_cache = make_cache(model, test, sorted(test_cold), image, text, args.batch_size, device)
    test_rows = prepare_rows(test_cache, locked)
    del test_cache
    ranks_original = original_ranks(test_rows)
    ranks_gated, target_alphas = gated_ranks(test_rows, **gate)
    original_metrics = metrics_from_ranks(ranks_original)
    gated_metrics = metrics_from_ranks(ranks_gated)

    gap_rows = semantic_gap_rows(args.data_dir, args.dataset)
    if len(gap_rows) != len(test_rows):
        raise RuntimeError("RQ3 semantic-gap rows do not align with the rebuilt test split")
    predictions = []
    for index, (row, gap, rank_old, rank_new, alpha) in enumerate(
        zip(test_rows, gap_rows, ranks_original, ranks_gated, target_alphas)
    ):
        sample = row["sample"]
        if sample.user != gap["user"] or sample.target != gap["target"]:
            raise RuntimeError(f"RQ3 semantic-gap identity mismatch at test row {index}")
        position = row["target_pos"]
        predictions.append({
            "row_id": index, "user": sample.user, "target": sample.target,
            "history_length": len(sample.prefix), "semantic_gap": gap["semantic_gap"],
            "gap_group": gap["gap_group"], "target_anchor": float(row["anchor"][position]),
            "target_anchor_percentile": float(row["anchor_percentile"][position]),
            "target_alpha": alpha, "rank_original": rank_old, "rank_gated": rank_new,
            "rank_improvement": rank_old - rank_new,
        })
    write_csv(output_dir / "gated_test_predictions.csv", predictions)
    by_gap = (
        metrics_by_gap(ranks_original, gap_rows, "FIR-MIC original")
        + metrics_by_gap(ranks_gated, gap_rows, "Gated-FIR-MIC")
    )
    write_csv(output_dir / "gated_test_by_gap.csv", by_gap)
    result = {
        "dataset": args.dataset, "seed": args.seed, "checkpoint": str(checkpoint_path),
        "device": str(device), "no_retraining": True, "selected_gate": gate,
        "selected_gate_satisfied_guardrails": bool(selected["eligible"]),
        "original": original_metrics, "gated": gated_metrics,
        "relative_gain": relative_gain(gated_metrics, original_metrics),
        "by_semantic_gap": by_gap, "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "gated_test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
