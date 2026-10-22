#!/usr/bin/env python3
"""FIR-MIC RQ3: balanced quartiles using 0.1 image + 0.9 text cosine.
Post-hoc analysis only: never trains or changes model checkpoints.
Run from the FIR-MIC-MMREC-Cold repository root.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

SEEDS = tuple(range(2022, 2032))
DATASETS = ("baby", "sports", "clothing")
METRICS = ("Recall@10", "Recall@20", "Recall@50", "Recall@100", "NDCG@20")

def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def normalized(a, idx):
    values = np.asarray(a[np.asarray(idx)], dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)

def resolve_data_dir(request):
    choices = ([request] if request else []) + [
        Path("/kaggle/input/datasets/toanktx/mmrec-cold"),
        Path("/kaggle/input/mmrec-cold"),
    ]
    for path in choices:
        if path and all((path / d / "image_feat.npy").is_file() and
                        (path / d / "text_feat.npy").is_file() for d in DATASETS):
            return path
    raise FileNotFoundError("Cannot locate MMRec arrays for baby/sports/clothing. "
                            "Attach Kaggle dataset toanktx/mmrec-cold, or pass --data-dir.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/rq3_01_09"))
    args = ap.parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path.cwd() / "acrg_apr_github"))
    from run_mmrec_seq_scl import Config, build_samples, read_interactions, temporal_item_split

    all_assignments, summary, by_seed = [], [], []
    for dataset in DATASETS:
        root = data_dir / dataset
        interactions = read_interactions(root, dataset)
        warm, val, test, cutoff, _ = temporal_item_split(interactions, Config(max_sequence_length=10))
        _, _, samples, _ = build_samples(interactions, warm, val, test, cutoff, 10)
        image = np.load(root / "image_feat.npy", mmap_mode="r")
        text = np.load(root / "text_feat.npy", mmap_mode="r")
        assignments = []
        for row_id, sample in enumerate(samples):
            history = list(sample.prefix)
            if not history:
                raise ValueError(f"Empty history in {dataset}: {row_id}")
            sim_v = normalized(image, history) @ normalized(image, [sample.target])[0]
            sim_t = normalized(text, history) @ normalized(text, [sample.target])[0]
            combined = 0.1 * sim_v + 0.9 * sim_t
            best = int(combined.argmax())
            assignments.append(dict(dataset=dataset, row_id=row_id, user=int(sample.user),
                target=int(sample.target), history_length=len(history),
                semantic_gap=1.0 - float(combined[best]),
                best_history_image_similarity=float(sim_v[best]),
                best_history_text_similarity=float(sim_t[best])))
        n = len(assignments)
        if not n:
            raise ValueError(f"No test samples: {dataset}")
        order = sorted(range(n), key=lambda index: (assignments[index]["semantic_gap"], assignments[index]["row_id"]))
        for pos, index in enumerate(order):
            assignments[index]["quartile"] = "Q" + str((pos * 4) // n + 1)
        lookup = {(str(a["user"]), str(a["target"])): a for a in assignments}
        if len(lookup) != n:
            raise ValueError(f"Duplicate (user,target) in {dataset}")
        all_assignments.extend(assignments)
        ranks_by_seed = {}
        for seed in SEEDS:
            path = args.results_dir / "fir_mic" / dataset / f"seed_{seed}" / "component_predictions.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            pred = read_csv(path)
            ranks = {}
            for row in pred:
                key = (str(int(row["user"])), str(int(row["target"])))
                if key in ranks or key not in lookup:
                    raise ValueError(f"Duplicate/unmatched prediction at {path}: {key}")
                ranks[key] = int(row["rank_FIR-MIC full"])
            if set(ranks) != set(lookup):
                raise ValueError(f"Missing {len(set(lookup)-set(ranks))} predictions in {path}")
            ranks_by_seed[seed] = ranks
        for q in ("Q1", "Q2", "Q3", "Q4"):
            group = [x for x in assignments if x["quartile"] == q]
            gaps = np.asarray([x["semantic_gap"] for x in group])
            per = []
            for seed in SEEDS:
                ranks = np.asarray([ranks_by_seed[seed][(str(x["user"]), str(x["target"]))] for x in group])
                if np.any(ranks < 1):
                    raise ValueError("Rank must be 1-based")
                row = dict(dataset=dataset, quartile=q, seed=seed, n=len(group))
                for k in (10,20,50,100):
                    row[f"Recall@{k}"] = float(np.mean(ranks <= k))
                row["NDCG@20"] = float(np.mean(np.where(ranks <= 20, 1.0/np.log2(ranks+1), 0)))
                by_seed.append(row)
                per.append(row)
            result = dict(dataset=dataset, quartile=q, n=len(group),
                g_min=float(gaps.min()), g_max=float(gaps.max()), g_mean=float(gaps.mean()))
            for metric in METRICS:
                values = np.asarray([r[metric] for r in per])
                result[metric+"_mean"] = float(values.mean())
                result[metric+"_sd"] = float(values.std(ddof=1))
            summary.append(result)
        print(f"{dataset}: {n} unique samples; 10 seeds validated; weight=0.1/0.9", flush=True)
    out = args.out_dir
    write_csv(out / "semantic_gap_assignments_01_09.csv", all_assignments)
    write_csv(out / "rq3_01_09_by_seed.csv", by_seed)
    write_csv(out / "rq3_01_09_summary.csv", summary)
    lines = [
        r"\begin{table*}[!t]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Khoảng cách ngữ nghĩa và chất lượng xếp hạng FIR-MIC theo các nhóm tứ phân vị (trọng số hình ảnh/văn bản $0.1/0.9$), trung bình 10 seed.}",
        r"\label{tab:rq3_semantic_gap}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrrrr@{}}",
        r"\toprule",
        r"Bộ dữ liệu & Nhóm & $n$ & $g_{\min}$ & $g_{\max}$ & $\overline{g}$ & R@10 & R@20 & R@50 & R@100 & N@20\\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        if dataset != "baby":
            lines.append(r"\midrule")
        for i, s in enumerate(r for r in summary if r["dataset"] == dataset):
            name = r"\multirow{4}{*}{" + dataset.title() + "}" if i == 0 else ""
            cells = [name, "$" + s["quartile"] + "$", str(s["n"])]
            cells += [format(s[k], ".4f") for k in ("g_min", "g_max", "g_mean")]
            cells += [format(s[k + "_mean"], ".5f") for k in METRICS]
            lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    out.mkdir(parents=True, exist_ok=True)
    (out/"rq3_01_09_table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Results written to {out.resolve()}", flush=True)

if __name__ == "__main__":
    main()
