"""Read an immutable local Git result snapshot; never tune on its test metrics."""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def audit(ref):
    def git(*args):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
            cwd=ROOT, text=True, encoding="utf8")
    commit = git("rev-parse", ref).strip()
    def read(path):
        return git("show", f"{commit}:{path}")
    def table(name):
        return list(csv.DictReader(io.StringIO(read(f"results/evidence/{name}.csv"))))
    paths = git("ls-tree", "-r", "--name-only", commit).splitlines()
    runs = [json.loads(read(path)) for path in paths
            if path.startswith("results/fir_mic/") and path.endswith("/result.json")]
    details = []
    for data in runs:
        config = data["optimal_config"]
        selected = config.get("selected_validation", {})
        final = config.get("confirmatory_validation", {})
        details.append({
            "dataset": data["dataset"], "seed": data["seed"],
            "intents": data["intents"], "epoch": data["epochs"],
            **data["validation_locked"], **data["route_diagnostics"],
            "test_users": data["test_users"],
            "mean_history": data.get("protocol_statistics", {}).get("average_history_length"),
            "selected_minus_confirmatory_ndcg10": selected.get("NDCG@10", 0) - final.get("NDCG@10", 0),
        })
    diagnostics = []
    for dataset in sorted({row["dataset"] for row in details}):
        rows = [row for row in details if row["dataset"] == dataset]
        diagnostics.append({
            "dataset": dataset, "n_seeds": len(rows),
            "router_disabled_seeds": sum(row["lambda_z"] == 0 for row in rows),
            "mean_predicted_stay": float(np.mean([row["mean_predicted_stay"] for row in rows])),
            "mean_target_stay": float(np.mean([row["mean_target_stay_similarity"] for row in rows])),
            "mean_route_auc": float(np.mean([row["median_split_auc"] for row in rows])),
            "max_selected_confirmatory_gap": max(abs(row["selected_minus_confirmatory_ndcg10"]) for row in rows),
            "test_user_counts": sorted({row["test_users"] for row in rows}),
        })
    return {"ref": ref, "commit": commit, "runs": len(runs),
            "main": table("main_mean_std"), "significance": table("statistical_significance"),
            "ablations": table("ablation_mean_std"), "diagnostics": diagnostics, "per_seed": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="origin/kaggle-results-gridsearch-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.ref)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({"commit": result["commit"], "runs": result["runs"],
                      "diagnostics": result["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
