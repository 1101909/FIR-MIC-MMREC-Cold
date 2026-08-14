"""Create paper-ready evidence tables from completed experiment artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = ("Recall@10", "NDCG@10", "MRR@10", "Recall@20", "NDCG@20", "MRR@20")
PRIMARY = "FIR-MIC"


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    # Evidence rows from different model families are intentionally sparse.
    # For example, only FIR-MIC reports hyperparameter-search time, while the
    # controlled baselines report other efficiency fields.  Preserve the
    # first-seen column order, but include fields found in every row.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def collect_main(results):
    rows = []
    controlled = results / "controlled_content" / "results_by_seed.csv"
    if controlled.exists():
        for row in read_csv(controlled):
            rows.append({"dataset": row["dataset"], "seed": int(row["seed"]), "model": row["model"],
                         **{metric: float(row[metric]) for metric in METRICS}})
    for path in results.glob("fir_mic/*/seed_*/result.json"):
        data = json.loads(path.read_text(encoding="utf8"))
        rows.append({"dataset": data["dataset"], "seed": int(data["seed"]), "model": PRIMARY,
                     **{metric: float(data["test"][metric]) for metric in METRICS}})
    for path in results.glob("official/*/seed_*.json"):
        data = json.loads(path.read_text(encoding="utf8"))
        for name, result in data["results"].items():
            # Legacy SEMCo artifacts used the optimistic all-tie rank rule and
            # have no diagnostics. Never mix them into paper-ready summaries.
            if name == "semco" and "diagnostics" not in result:
                continue
            rows.append({"dataset": data["dataset"], "seed": int(data["seed"]), "model": name.upper(),
                         **{metric: float(result["test"][metric]) for metric in METRICS}})
    return rows


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows: groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for group, values in sorted(groups.items(), key=lambda pair: tuple(map(str, pair[0]))):
        row = dict(zip(keys, group)); row["n_seeds"] = len(values)
        for metric in METRICS:
            mean, sd = mean_sd([value[metric] for value in values])
            row[f"{metric}_mean"] = mean; row[f"{metric}_sd"] = sd
        output.append(row)
    return output


def gains(summary):
    output = []
    by_dataset = defaultdict(list)
    for row in summary: by_dataset[row["dataset"]].append(row)
    for dataset, rows in by_dataset.items():
        proposed = next((row for row in rows if row["model"] == PRIMARY), None)
        if not proposed: continue
        valid_baselines = [row for row in rows if row["model"] != PRIMARY]
        strongest = max(valid_baselines, key=lambda row: row["NDCG@10_mean"])["model"] if valid_baselines else None
        for baseline in valid_baselines:
            row = {"dataset": dataset, "baseline": baseline["model"],
                   "strongest_baseline": int(baseline["model"] == strongest)}
            for metric in METRICS:
                base = baseline[f"{metric}_mean"]
                row[f"gain_pct_{metric}"] = 100 * (proposed[f"{metric}_mean"] - base) / base if base else float("nan")
            output.append(row)
    return output


def holm(pvalues):
    order = np.argsort(pvalues); adjusted = np.empty(len(pvalues)); running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value); adjusted[index] = running
    return adjusted


def paired_significance(raw_rows, baseline):
    index = {(row["dataset"], row["seed"], row["model"]): row for row in raw_rows}
    tests = []
    datasets = sorted({row["dataset"] for row in raw_rows})
    for dataset in datasets:
        seeds = sorted({row["seed"] for row in raw_rows if row["dataset"] == dataset})
        for metric in METRICS:
            pairs = [(index[(dataset, seed, PRIMARY)][metric], index[(dataset, seed, baseline)][metric])
                     for seed in seeds if (dataset, seed, PRIMARY) in index and (dataset, seed, baseline) in index]
            if len(pairs) < 2: continue
            proposed, reference = map(np.asarray, zip(*pairs)); delta = proposed - reference
            mean = float(delta.mean()); sd = float(delta.std(ddof=1)); se = sd / math.sqrt(len(delta))
            half = float(stats.t.ppf(0.975, len(delta) - 1) * se)
            try: wilcoxon_p = float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
            except ValueError: wilcoxon_p = 1.0
            pooled = math.sqrt((proposed.var(ddof=1) + reference.var(ddof=1)) / 2)
            tests.append({
                "dataset": dataset, "baseline": baseline, "metric": metric, "n_seeds": len(delta),
                "mean_paired_difference": mean, "relative_improvement_pct": 100 * mean / reference.mean(),
                "std_paired_difference": sd, "ci95_low": mean - half, "ci95_high": mean + half,
                "paired_t_p": float(stats.ttest_rel(proposed, reference).pvalue),
                "wilcoxon_p": wilcoxon_p, "effect_size_cohens_dz": mean / sd if sd else float("inf"),
                "effect_size_pooled_sd": mean / pooled if pooled else float("inf"),
            })
    if tests:
        for field in ("paired_t_p", "wilcoxon_p"):
            adjusted = holm([row[field] for row in tests])
            for row, value in zip(tests, adjusted): row[f"holm_adjusted_{field}"] = float(value)
    return tests


def collect_json_stats(results, filename, section):
    rows = []
    for path in results.glob(f"fir_mic/*/seed_*/{filename}"):
        data = json.loads(path.read_text(encoding="utf8"))
        rows.append({"dataset": path.parents[1].name, "seed": int(path.parent.name.split("_")[-1]), **data})
    if not rows: return []
    keys = [key for key in rows[0] if key not in ("dataset", "seed")]
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset]
        summary = {"dataset": dataset, "n_seeds": len(selected), "section": section}
        for key in keys:
            values = [row[key] for row in selected if isinstance(row[key], (int, float))]
            if values:
                summary[f"{key}_mean"], summary[f"{key}_sd"] = mean_sd(values)
        output.append(summary)
    return output


def collect_all_efficiency(results):
    rows = []
    controlled = results / "controlled_content" / "results_by_seed.csv"
    if controlled.exists():
        for row in read_csv(controlled):
            rows.append({"dataset": row["dataset"], "seed": row["seed"], "model": row["model"],
                         **{field: float(row[field]) for field in (
                             "train_seconds", "inference_ms_per_user", "trainable_parameters",
                             "peak_cuda_memory_mb")}})
    for path in results.glob("fir_mic/*/seed_*/efficiency.json"):
        data = json.loads(path.read_text(encoding="utf8"))
        rows.append({"dataset": path.parents[1].name, "seed": path.parent.name.split("_")[-1],
                     "model": PRIMARY, **data})
    for path in results.glob("official/*/seed_*.json"):
        data = json.loads(path.read_text(encoding="utf8"))
        for model, result in data["results"].items():
            if "efficiency" in result:
                rows.append({"dataset": data["dataset"], "seed": data["seed"],
                             "model": model.upper(), **result["efficiency"]})
    if not rows: return []
    groups = defaultdict(list)
    for row in rows: groups[(row["dataset"], row["model"])].append(row)
    output = []
    for (dataset, model), values in sorted(groups.items()):
        summary = {"dataset": dataset, "model": model, "n_seeds": len(values)}
        fields = sorted(set().union(*(value.keys() for value in values)) - {"dataset", "seed", "model"})
        for field in fields:
            numeric = [float(value[field]) for value in values if field in value]
            summary[f"{field}_mean"], summary[f"{field}_sd"] = mean_sd(numeric)
        output.append(summary)
    return output


def collect_ablations(results):
    rows = []
    for path in results.glob("fir_mic/*/seed_*/component_ablation.csv"):
        dataset, seed = path.parents[1].name, int(path.parent.name.split("_")[-1])
        rows += [{"dataset": dataset, "seed": seed, "variant": row["variant"],
                  **{metric: float(row[metric]) for metric in METRICS}} for row in read_csv(path)]
    for path in results.glob("fir_mic/*/seed_*/modality_ablation.csv"):
        dataset, seed = path.parents[1].name, int(path.parent.name.split("_")[-1])
        for row in read_csv(path):
            if row["variant"] != "FIR-MIC full":
                rows.append({"dataset": dataset, "seed": seed, "variant": row["variant"],
                             **{metric: float(row[metric]) for metric in METRICS}})
    return aggregate(rows, ("dataset", "variant")) if rows else []


def collect_hyperparameters(results):
    selected, trials = [], []
    for path in results.glob("fir_mic/*/seed_*/optimal_config.json"):
        dataset = path.parents[1].name
        seed = int(path.parent.name.split("_")[-1])
        data = json.loads(path.read_text(encoding="utf8"))
        row = {
            "dataset": dataset, "seed": seed,
            **{key: data[key] for key in (
                "trial", "intents", "dim", "learning_rate", "weight_decay",
                "dropout", "router_dropout", "best_epoch", "search_max_epochs",
                "grid_patience", "search_seconds", "completed_trials", "total_trials",
            ) if key in data},
            "selection_split": data.get("selection_split"),
            "selection_metric": data.get("selection_metric"),
        }
        row.update({f"selected_validation_{key}": value
                    for key, value in data.get("selected_validation", {}).items()})
        row.update({f"confirmatory_validation_{key}": value
                    for key, value in data.get("confirmatory_validation", {}).items()})
        row.update({f"locked_{key}": value
                    for key, value in data.get("locked_score_weights", {}).items()})
        row["search_space"] = json.dumps(data.get("search_space", {}), sort_keys=True)
        selected.append(row)

        trial_path = path.parent / "hyperparameter_trials.csv"
        if trial_path.exists():
            trials.extend({"dataset": dataset, "seed": seed, **trial}
                          for trial in read_csv(trial_path))
    return selected, trials


def aggregate_numeric_table(path, keys):
    if not path.exists(): return []
    source = read_csv(path); groups = defaultdict(list)
    for row in source: groups[tuple(row[key] for key in keys)].append(row)
    output = []
    excluded = set(keys) | {"seed"}
    for group, rows in sorted(groups.items(), key=lambda pair: tuple(map(str, pair[0]))):
        summary = dict(zip(keys, group)); summary["n_seeds"] = len({row.get("seed") for row in rows})
        for field in rows[0]:
            if field in excluded: continue
            try: values = [float(row[field]) for row in rows if row[field] != ""]
            except (TypeError, ValueError): continue
            if values:
                summary[f"{field}_mean"], summary[f"{field}_sd"] = mean_sd(values)
        output.append(summary)
    return output


def collect_rq3_tables(results, output):
    specifications = {
        "past_future_by_seed.csv": ("past_future_mean_std.csv", ("dataset", "scope", "variant")),
        "future_state_quality_by_seed.csv": ("future_state_quality_mean_std.csv", ("dataset", "scope", "measure")),
        "history_analysis_by_seed.csv": ("history_analysis_mean_std.csv", ("dataset", "group", "model")),
        "diversity_analysis_by_seed.csv": ("diversity_analysis_mean_std.csv", ("dataset", "group", "model")),
        "history_gain_by_seed.csv": ("history_gain_mean_std.csv", ("dataset", "group")),
        "diversity_gain_by_seed.csv": ("diversity_gain_mean_std.csv", ("dataset", "group")),
    }
    for source_name, (output_name, keys) in specifications.items():
        rows = []
        for path in results.glob(f"rq3/*/{source_name}"):
            rows += aggregate_numeric_table(path, keys)
        write_csv(output / output_name, rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(); output = args.results_dir / "evidence"; output.mkdir(parents=True, exist_ok=True)
    raw = collect_main(args.results_dir); summary = aggregate(raw, ("dataset", "model"))
    write_csv(output / "main_raw_by_seed.csv", raw)
    write_csv(output / "main_mean_std.csv", summary)
    gain_rows = gains(summary); write_csv(output / "main_gains.csv", gain_rows)
    strongest = {row["dataset"]: row["baseline"] for row in gain_rows if row["strongest_baseline"]}
    significance = []
    for baseline in sorted(set(strongest.values())): significance += paired_significance(raw, baseline)
    significance = [row for row in significance if strongest.get(row["dataset"]) == row["baseline"]]
    if significance:
        for field in ("paired_t_p", "wilcoxon_p"):
            adjusted = holm([row[field] for row in significance])
            for row, value in zip(significance, adjusted):
                row[f"holm_adjusted_{field}"] = float(value)
    write_csv(output / "statistical_significance.csv", significance)
    ablations = collect_ablations(args.results_dir)
    for row in summary:
        label = {"Static score-level": "Static", "Multi-interest per-item": "Per-item"}.get(row["model"])
        if label: ablations.append({"dataset": row["dataset"], "variant": label,
                                    **{key: value for key, value in row.items() if key not in ("dataset", "model")}})
    write_csv(output / "ablation_mean_std.csv", ablations)
    write_csv(output / "protocol_mean_std.csv", collect_json_stats(args.results_dir, "protocol_statistics.json", "protocol"))
    write_csv(output / "transition_mean_std.csv", collect_json_stats(args.results_dir, "transition_statistics.json", "transition"))
    write_csv(output / "efficiency_mean_std.csv", collect_all_efficiency(args.results_dir))
    selected_configs, search_trials = collect_hyperparameters(args.results_dir)
    write_csv(output / "optimal_configs_by_seed.csv", selected_configs)
    write_csv(output / "hyperparameter_trials_by_seed.csv", search_trials)
    collect_rq3_tables(args.results_dir, output)
    print(json.dumps({"datasets": sorted({row['dataset'] for row in raw}), "raw_rows": len(raw),
                      "output": str(output)}, indent=2))


if __name__ == "__main__": main()
