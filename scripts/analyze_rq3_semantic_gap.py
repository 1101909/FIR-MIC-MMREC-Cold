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
from scipy import stats

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
        if len(history) >= 2:
            history_similarity = 0.2 * (image_history @ image_history.T) + 0.8 * (text_history @ text_history.T)
            upper = history_similarity[np.triu_indices(len(history), k=1)]
            diversity = 1.0 - float(upper.mean())
        else:
            diversity = None
        length = len(history)
        rows.append({
            "row_id": row_id,
            "user": sample.user,
            "target": sample.target,
            "prefix_length": length,
            "history_group": "1" if length == 1 else "2-3" if length <= 3 else "4-5" if length <= 5 else ">=6",
            "interest_diversity": diversity,
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
    diverse = sorted(
        (row for row in rows if row["interest_diversity"] is not None),
        key=lambda row: (row["interest_diversity"], row["row_id"]),
    )
    for position, row in enumerate(diverse):
        level = min(3, position * 3 // len(diverse) + 1)
        row["diversity_group"] = ("Low", "Medium", "High")[level - 1]
    for row in rows:
        if row["interest_diversity"] is None:
            row["diversity_group"] = "Single-history"
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
                row[f"gain_pct_{metric}"] = (
                    100.0 * row[f"delta_{metric}"] / float(baseline[metric])
                    if float(baseline[metric]) != 0 else float("nan")
                )
            rows.append(row)
    return rows


def rq3_report(effect_rows, metric_rows, seeds_analyzed):
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
        "high_minus_overall_interaction": {},
        "high_minus_overall_relative_gain": {},
        "overall_effect": {},
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
        metric_index = defaultdict(dict)
        for row in metric_rows:
            metric_index[(int(row["seed"]), row["model"])][int(row["gap_quartile"])] = row
        overall_deltas = []
        overall_gain_pct = []
        for seed in sorted({int(row["seed"]) for row in metric_rows}):
            proposed = metric_index.get((seed, PRIMARY_MODEL), {})
            baseline = metric_index.get((seed, PRIMARY_BASELINE), {})
            if len(proposed) == 4 and len(baseline) == 4:
                p = np.average([proposed[q][metric] for q in range(1, 5)],
                               weights=[proposed[q]["n_users"] for q in range(1, 5)])
                b = np.average([baseline[q][metric] for q in range(1, 5)],
                               weights=[baseline[q]["n_users"] for q in range(1, 5)])
                overall_deltas.append(float(p - b))
                overall_gain_pct.append(float(100 * (p - b) / b) if b else float("nan"))
        report["overall_effect"][metric] = {
            "mean_delta": float(np.mean(overall_deltas)) if overall_deltas else float("nan"),
            "exact_sign_flip_p": exact_sign_flip_p(overall_deltas),
        }
        contrasts = [
            groups[4][f"delta_{metric}"] - groups[1][f"delta_{metric}"]
            for groups in paired.values() if 1 in groups and 4 in groups
        ]
        report["high_minus_low_interaction"][metric] = {
            "mean_delta_difference": float(np.mean(contrasts)) if contrasts else float("nan"),
            "exact_sign_flip_p": exact_sign_flip_p(contrasts),
            "supports_larger_high_gap_gain": bool(contrasts and np.mean(contrasts) > 0),
        }
        q4 = [groups[4][f"delta_{metric}"] for groups in paired.values() if 4 in groups]
        high_overall = [high - overall for high, overall in zip(q4, overall_deltas)]
        report["high_minus_overall_interaction"][metric] = {
            "mean_delta_difference": float(np.mean(high_overall)) if high_overall else float("nan"),
            "exact_sign_flip_p": exact_sign_flip_p(high_overall),
            "supports_larger_high_gap_gain": bool(high_overall and np.mean(high_overall) > 0),
        }
        q4_gain_pct = [groups[4][f"gain_pct_{metric}"] for groups in paired.values() if 4 in groups]
        relative_contrast = [high - overall for high, overall in zip(q4_gain_pct, overall_gain_pct)]
        report["high_minus_overall_relative_gain"][metric] = {
            "overall_gain_pct": float(np.mean(overall_gain_pct)) if overall_gain_pct else float("nan"),
            "q4_gain_pct": float(np.mean(q4_gain_pct)) if q4_gain_pct else float("nan"),
            "q4_minus_overall_gain_percentage_points": (
                float(np.mean(relative_contrast)) if relative_contrast else float("nan")
            ),
            "exact_sign_flip_p": exact_sign_flip_p(relative_contrast),
        }
    return report


def confidence_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    half = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1) * values.std(ddof=1) / math.sqrt(len(values)))
    return mean - half, mean + half


def diagnostic_evidence(results_dir, dataset, assignment_index, output_dir):
    past_future_rows, quality_rows = [], []
    history_rows, diversity_rows, history_gain_rows, diversity_gain_rows = [], [], [], []
    component_root = results_dir / "fir_mic" / dataset
    for path in sorted(component_root.glob("seed_*/component_predictions.csv")):
        seed = int(path.parent.name.split("_")[-1])
        components = {
            (int(row["user"]), int(row["target"])): row for row in read_csv(path)
        }
        per_item_path = results_dir / "controlled_content" / MODELS[PRIMARY_BASELINE].format(
            dataset=dataset, seed=seed
        )
        if set(components) != set(assignment_index) or not per_item_path.exists():
            raise AssertionError(f"Diagnostic keys or per-item predictions are incomplete for seed {seed}")
        per_item = load_ranks(per_item_path, PRIMARY_BASELINE)

        for scope, selected in (
            ("Overall", list(assignment_index)),
            ("Q4-high", [key for key, row in assignment_index.items() if row["gap_quartile"] == 4]),
        ):
            for variant in ("Past-only", "Future-only", "Past+Future", "FIR-MIC full"):
                ranks = [int(components[key][f"rank_{variant}"]) for key in selected]
                past_future_rows.append({
                    "dataset": dataset, "seed": seed, "scope": scope,
                    "variant": variant, "n_users": len(selected), **ranking_metrics(ranks),
                })
            past = np.asarray([float(components[key]["past_target_cosine"]) for key in selected])
            future = np.asarray([float(components[key]["future_target_cosine"]) for key in selected])
            delta = future - past
            for name, values in (("past_target_cosine", past), ("future_target_cosine", future),
                                 ("delta_future_minus_past", delta)):
                low, high = confidence_interval(values)
                quality_rows.append({
                    "dataset": dataset, "seed": seed, "scope": scope, "measure": name,
                    "n_users": len(values), "mean": float(values.mean()),
                    "median": float(np.median(values)), "std": float(values.std(ddof=1)),
                    "ci95_low": low, "ci95_high": high,
                    "pct_future_better": float(np.mean(delta > 0)) if name == "delta_future_minus_past" else "",
                })

        full_ranks = {key: int(row["rank_FIR-MIC full"]) for key, row in components.items()}
        for field, destination, gain_destination in (
            ("history_group", history_rows, history_gain_rows),
            ("diversity_group", diversity_rows, diversity_gain_rows),
        ):
            groups = sorted({row[field] for row in assignment_index.values()})
            for group in groups:
                selected = [key for key, row in assignment_index.items() if row[field] == group]
                if not selected:
                    continue
                avg_history = float(np.mean([assignment_index[key]["prefix_length"] for key in selected]))
                avg_diversity_values = [assignment_index[key]["interest_diversity"] for key in selected
                                        if assignment_index[key]["interest_diversity"] is not None]
                common = {
                    "dataset": dataset, "seed": seed, "group": group,
                    "n_users": len(selected), "average_history_length": avg_history,
                    "average_interest_diversity": (
                        float(np.mean(avg_diversity_values)) if avg_diversity_values else ""
                    ),
                }
                base_metrics = ranking_metrics([per_item[key] for key in selected])
                proposed_metrics = ranking_metrics([full_ranks[key] for key in selected])
                destination.append({**common, "model": PRIMARY_BASELINE, **base_metrics})
                destination.append({**common, "model": PRIMARY_MODEL, **proposed_metrics})
                gain = {f"gain_pct_{metric}": (
                    100 * (proposed_metrics[metric] - base_metrics[metric]) / base_metrics[metric]
                    if base_metrics[metric] else float("nan")
                ) for metric in METRICS}
                gain_destination.append({**common, **gain})

    write_csv(output_dir / "past_future_by_seed.csv", past_future_rows)
    write_csv(output_dir / "future_state_quality_by_seed.csv", quality_rows)
    write_csv(output_dir / "history_analysis_by_seed.csv", history_rows)
    write_csv(output_dir / "diversity_analysis_by_seed.csv", diversity_rows)
    write_csv(output_dir / "history_gain_by_seed.csv", history_gain_rows)
    write_csv(output_dir / "diversity_gain_by_seed.csv", diversity_gain_rows)


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
    report = rq3_report(effects, metric_rows, sorted(seeds))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "metrics_by_seed.csv", metric_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_fir_mic_vs_per_item.csv", effects)
    diagnostic_evidence(args.results_dir, args.dataset, assignment_index, output_dir)
    (output_dir / "rq3_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
