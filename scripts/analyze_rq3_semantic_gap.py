"""RQ3: does FIR-MIC help more when history and target semantics diverge?

The semantic-gap definition is fixed from the earlier hierarchy pilot:

    1 - max_{i in history} [0.2 cos(image_i, image_target)
                            + 0.8 cos(text_i, text_target)]

Gap groups depend only on intrinsic content and the leakage-safe test prefix;
they never inspect a model score, rank, or cold interaction during training.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_mmrec_seq_scl import Config, build_samples, read_interactions, temporal_item_split


MODELS = {
    "Static score-level": "{dataset}_static_score-level_seed_{seed}_predictions.csv",
    "Multi-interest per-item": "{dataset}_multi-interest_per-item_seed_{seed}_predictions.csv",
    "Multi-interest clustered": "{dataset}_multi-interest_clustered_seed_{seed}_predictions.csv",
}
PRIMARY_BASELINE = "Multi-interest per-item"
PRIMARY_MODEL = "FIR-MIC"
METRICS = ("Recall@10", "NDCG@10", "MRR@10", "Recall@20", "NDCG@20", "MRR@20")


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_rows(array, indices):
    rows = np.asarray(array[np.asarray(indices)], dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-9)


def semantic_gap_rows(data_dir: Path, dataset: str):
    root = data_dir / dataset
    interactions = read_interactions(root, dataset)
    warm, validation, test, validation_start, _ = temporal_item_split(
        interactions, Config(max_sequence_length=10)
    )
    _, _, samples, _ = build_samples(
        interactions, warm, validation, test, validation_start, 10
    )
    image = np.load(root / "image_feat.npy", mmap_mode="r")
    text = np.load(root / "text_feat.npy", mmap_mode="r")
    rows = []
    for row_id, sample in enumerate(samples):
        history = list(sample.prefix)
        image_history = normalized_rows(image, history)
        text_history = normalized_rows(text, history)
        image_target = normalized_rows(image, [sample.target])[0]
        text_target = normalized_rows(text, [sample.target])[0]
        image_similarity = image_history @ image_target
        text_similarity = text_history @ text_target
        combined = 0.2 * image_similarity + 0.8 * text_similarity
        best = int(np.argmax(combined))
        rows.append({
            "row_id": row_id,
            "user": sample.user,
            "target": sample.target,
            "prefix_length": len(history),
            "semantic_gap": 1.0 - float(combined[best]),
            "best_history_image_similarity": float(image_similarity[best]),
            "best_history_text_similarity": float(text_similarity[best]),
        })

    # Rank-balanced, deterministic quartiles avoid ambiguous quantile edges
    # when several samples have exactly the same content similarity.
    ordered = sorted(rows, key=lambda row: (row["semantic_gap"], row["row_id"]))
    for position, row in enumerate(ordered):
        quartile = min(4, position * 4 // len(ordered) + 1)
        row["gap_quartile"] = quartile
        row["gap_group"] = ("Q1-low", "Q2", "Q3", "Q4-high")[quartile - 1]
    return sorted(rows, key=lambda row: row["row_id"])


def prediction_paths(results_dir: Path, dataset: str):
    controlled = results_dir / "controlled_content"
    found = []
    for model, pattern in MODELS.items():
        regex = re.compile(re.escape(pattern.format(dataset=dataset, seed=0)).replace("0", r"(\d+)") + "$" )
        for path in controlled.glob(pattern.format(dataset=dataset, seed="*")):
            match = regex.search(path.name)
            if match:
                found.append((model, int(match.group(1)), path))
    for path in (results_dir / "fir_mic" / dataset).glob("seed_*/predictions.csv"):
        found.append((PRIMARY_MODEL, int(path.parent.name.split("_")[-1]), path))
    return sorted(found, key=lambda row: (row[1], row[0]))


def load_ranks(path: Path, model: str):
    rows = read_csv(path)
    if model == PRIMARY_MODEL:
        return {(int(row["user"]), int(row["target"])): int(row["rank"]) for row in rows}
    return {
        (int(row["user_id"]), int(row["target_item"])): int(row["rank"])
        for row in rows
    }


def ranking_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    output = {}
    for k in (10, 20):
        output[f"Recall@{k}"] = float(np.mean(ranks <= k))
        output[f"NDCG@{k}"] = float(np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)))
        output[f"MRR@{k}"] = float(np.mean(np.where(ranks <= k, 1 / ranks, 0)))
    return output


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, values in sorted(groups.items(), key=lambda pair: tuple(map(str, pair[0]))):
        row = dict(zip(keys, group))
        row["n_seeds"] = len(values)
        row["n_users"] = int(round(np.mean([int(value["n_users"]) for value in values])))
        row["mean_gap"] = float(np.mean([float(value["mean_gap"]) for value in values]))
        for metric in METRICS:
            data = np.asarray([float(value[metric]) for value in values])
            row[f"{metric}_mean"] = float(data.mean())
            row[f"{metric}_sd"] = float(data.std(ddof=1)) if len(data) > 1 else 0.0
        output.append(row)
    return output


def exact_sign_flip_p(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        extreme += abs(float(np.mean(values * signs))) >= observed - 1e-15
    return extreme / total


def paired_effects(metric_rows):
    index = {
        (int(row["seed"]), int(row["gap_quartile"]), row["model"]): row
        for row in metric_rows
    }
    seeds = sorted({key[0] for key in index})
    rows = []
    for seed in seeds:
        for quartile in range(1, 5):
            proposed = index.get((seed, quartile, PRIMARY_MODEL))
            baseline = index.get((seed, quartile, PRIMARY_BASELINE))
            if not proposed or not baseline:
                continue
            row = {"seed": seed, "gap_quartile": quartile}
            for metric in METRICS:
                row[f"delta_{metric}"] = float(proposed[metric]) - float(baseline[metric])
            rows.append(row)
    return rows


def rq3_report(effect_rows, seeds_analyzed):
    by_quartile = defaultdict(list)
    for row in effect_rows:
        by_quartile[int(row["gap_quartile"])].append(row)
    report = {
        "research_question": "RQ3: Does FIR-MIC improve more as history-to-target semantic gap increases?",
        "semantic_gap": "1 - max_history(0.2*cos(image) + 0.8*cos(text))",
        "groups": "deterministic rank-balanced quartiles; Q1=lowest gap, Q4=highest gap",
        "primary_comparison": f"{PRIMARY_MODEL} minus {PRIMARY_BASELINE}",
        "seeds_analyzed": seeds_analyzed,
        "quartile_effects": {},
        "high_minus_low_interaction": {},
    }
    for quartile in range(1, 5):
        values = by_quartile.get(quartile, [])
        report["quartile_effects"][f"Q{quartile}"] = {
            metric: {
                "mean_delta": float(np.mean([row[f"delta_{metric}"] for row in values])),
                "exact_sign_flip_p": exact_sign_flip_p([row[f"delta_{metric}"] for row in values]),
            }
            for metric in METRICS
        } if values else {}
    paired = defaultdict(dict)
    for row in effect_rows:
        paired[int(row["seed"])][int(row["gap_quartile"])] = row
    for metric in METRICS:
        contrasts = [
            groups[4][f"delta_{metric}"] - groups[1][f"delta_{metric}"]
            for groups in paired.values() if 1 in groups and 4 in groups
        ]
        report["high_minus_low_interaction"][metric] = {
            "mean_delta_difference": float(np.mean(contrasts)) if contrasts else float("nan"),
            "exact_sign_flip_p": exact_sign_flip_p(contrasts),
            "supports_larger_high_gap_gain": bool(contrasts and np.mean(contrasts) > 0),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("baby", "clothing", "sports"), required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    assignments = semantic_gap_rows(args.data_dir, args.dataset)
    assignment_index = {(row["user"], row["target"]): row for row in assignments}
    output_dir = args.results_dir / "rq3" / args.dataset
    write_csv(output_dir / "semantic_gap_assignments.csv", assignments)

    metric_rows = []
    seeds = set()
    for model, seed, path in prediction_paths(args.results_dir, args.dataset):
        ranks = load_ranks(path, model)
        if set(ranks) != set(assignment_index):
            raise AssertionError(
                f"{path} does not match the strict test sample set: "
                f"predictions={len(ranks)}, expected={len(assignment_index)}"
            )
        seeds.add(seed)
        for quartile in range(1, 5):
            selected = [
                key for key, row in assignment_index.items()
                if row["gap_quartile"] == quartile
            ]
            values = [ranks[key] for key in selected]
            metric_rows.append({
                "dataset": args.dataset, "seed": seed, "model": model,
                "gap_quartile": quartile,
                "gap_group": ("Q1-low", "Q2", "Q3", "Q4-high")[quartile - 1],
                "n_users": len(selected),
                "mean_gap": float(np.mean([assignment_index[key]["semantic_gap"] for key in selected])),
                **ranking_metrics(values),
            })

    effects = paired_effects(metric_rows)
    summary = summarize(metric_rows, ("dataset", "model", "gap_quartile", "gap_group"))
    report = rq3_report(effects, sorted(seeds))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics_by_seed.csv", metric_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_fir_mic_vs_per_item.csv", effects)
    (output_dir / "rq3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
