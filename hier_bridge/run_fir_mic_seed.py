"""FIR-MIC: future-interest routed multi-interest cold recommendation.

CPU-friendly research runner.  It preserves the strong item-wise content
anchor, learns soft interest tokens from all original feature coordinates, and
trains a stay/switch router on warm-only pseudo-cold targets.  Cold interactions
are never used for representation learning or model selection.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GRID_SEARCH_VERSION = 1
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "acrg_apr_github"))

from run_intent_interpolation_seed import intent_space, l2_blocks
from run_mmrec_seq_scl import Config, build_samples, leakage_checks, read_interactions, temporal_item_split
from run_soft_intent_bridge_v2_seed import (
    DS,
    Model,
    collate,
    pseudo_cold,
    raw_anchor,
    warm_teacher,
)


class FIRMIC(Model):
    def __init__(self, image, text, teacher, k, d=64, maxlen=10, modality="full",
                 dropout=0.15, router_dropout=0.10):
        super().__init__(image, text, teacher, k, d, maxlen, modality=modality, dropout=dropout)
        self.router = torch.nn.Sequential(
            torch.nn.Linear(5, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(router_dropout),
            torch.nn.Linear(16, 1),
        )
        self.router_dropout = router_dropout

    def routed_components(self, p, lengths, candidates):
        current, future, history, mask, q_hist = self.encode(p, lengths)
        q_c = self.item_q(candidates)
        cold = self.content(candidates)

        stay = current @ q_c.T
        delta = F.relu(future - current)
        delta = delta / delta.sum(-1, keepdim=True).clamp_min(1e-6)
        switch = delta @ q_c.T
        next_score = future @ q_c.T

        entropy_f = -(future * torch.log(future + 1e-9)).sum(-1) / math.log(self.k)
        entropy_c = -(current * torch.log(current + 1e-9)).sum(-1) / math.log(self.k)
        cosine = F.cosine_similarity(current, future)
        shift = (future - current).abs().sum(-1) / 2
        if q_hist.shape[1] > 1:
            valid_pair = (~mask[:, 1:]) & (~mask[:, :-1])
            pair_sim = (q_hist[:, 1:] * q_hist[:, :-1]).sum(-1)
            repetition = (pair_sim * valid_pair).sum(-1) / valid_pair.sum(-1).clamp_min(1)
        else:
            repetition = torch.ones_like(cosine)
        features = torch.stack([cosine, entropy_f, entropy_c, shift, repetition], -1)
        route = torch.sigmoid(self.router(features)).squeeze(-1)
        routed = route[:, None] * stay + (1 - route[:, None]) * switch

        att = torch.einsum("bld,cd->blc", history, cold)
        att = att.masked_fill(mask[:, :, None], -1e9).softmax(1)
        context = torch.einsum("blc,bld->bcd", att, history)
        fine = torch.einsum("bcd,cd->bc", context, cold)
        return current, future, route, stay, switch, routed, next_score, fine, q_c


def row_z(x):
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-5)


def train_model(model, samples, image, text, epochs, batch_size, seed, device,
                learning_rate=1e-3, weight_decay=1e-4, epoch_callback=None,
                patience=None):
    loader = DataLoader(
        DS(samples), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    best_key = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        totals, ranks, next_losses, route_losses = [], [], [], []
        for p, lengths, target, _ in loader:
            p, lengths, target = p.to(device), lengths.to(device), target.to(device)
            optimizer.zero_grad()
            current, future, route, _, _, routed, next_score, fine, q_target = \
                model.routed_components(p, lengths, target)
            anchor = raw_anchor(p, lengths, target, image, text, model.modality)
            fine_residual = fine - anchor
            score = row_z(anchor) + .10 * row_z(routed) + .10 * row_z(next_score) + .20 * row_z(fine_residual)
            labels = torch.arange(len(target), device=device)
            rank_loss = F.cross_entropy(score / .10, labels)

            target_q = q_target.detach()
            next_loss = -(target_q * torch.log(future + 1e-9)).sum(-1).mean()
            stay_label = (current.detach() * target_q).sum(-1).clamp(0, 1)
            route_loss = F.binary_cross_entropy(route, stay_label)
            teacher_loss = F.kl_div(
                torch.log(model.item_q(target) + 1e-9), model.teacher[target], reduction="batchmean"
            )
            mean_q = model.item_q(target).mean(0)
            balance = (mean_q * torch.log(mean_q * model.k + 1e-9)).sum()
            loss = rank_loss + .20 * next_loss + .10 * route_loss + .05 * teacher_loss + .01 * balance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            totals.append(float(loss)); ranks.append(float(rank_loss))
            next_losses.append(float(next_loss)); route_losses.append(float(route_loss))
        row = {
            "epoch": epoch, "loss": float(np.mean(totals)),
            "rank_loss": float(np.mean(ranks)), "next_loss": float(np.mean(next_losses)),
            "route_loss": float(np.mean(route_losses)),
        }
        if epoch_callback is not None:
            validation = epoch_callback(model, epoch)
            row.update({f"validation_{key}": value for key, value in validation.items()})
            key = (
                validation["NDCG@10"], validation["Recall@10"],
                validation["MRR@10"],
            )
            if best_key is None or key > best_key:
                best_key = key
                stale_epochs = 0
            else:
                stale_epochs += 1
        history.append(row)
        print(json.dumps(row), flush=True)
        if patience and stale_epochs >= patience:
            break
    return history


@torch.no_grad()
def make_cache(model, samples, candidates, image, text, batch_size, device):
    model.eval()
    candidate_tensor = torch.tensor(candidates, device=device)
    lookup = {item: pos for pos, item in enumerate(candidates)}
    rows = []
    loader = DataLoader(DS(samples), batch_size=batch_size, collate_fn=collate)
    for p, lengths, _, source_rows in loader:
        p, lengths = p.to(device), lengths.to(device)
        current, future, route, stay, switch, routed, next_score, fine, q_c = \
            model.routed_components(p, lengths, candidate_tensor)
        anchor = raw_anchor(p, lengths, candidate_tensor, image, text, model.modality)
        for j, source in enumerate(source_rows):
            pos = lookup[source.target]
            true_stay = float((current[j] * q_c[pos]).sum().cpu())
            rows.append({
                "sample": source, "target_pos": pos,
                "anchor": anchor[j].cpu().numpy(),
                "stay": stay[j].cpu().numpy(),
                "switch": switch[j].cpu().numpy(),
                "routed": routed[j].cpu().numpy(),
                "next": next_score[j].cpu().numpy(),
                "fine_residual": (fine[j] - anchor[j]).cpu().numpy(),
                "current": current[j].cpu().numpy(),
                "future": future[j].cpu().numpy(),
                "target_q": q_c[pos].cpu().numpy(),
                "route": float(route[j].cpu()), "true_stay": true_stay,
            })
    return rows


def z(x):
    return (x - x.mean()) / max(float(x.std()), 1e-6)


def stable_rank(score, target_position):
    order = np.argsort(-score, kind="stable")
    return int(np.flatnonzero(order == target_position)[0] + 1)


def metrics_from_ranks(ranks):
    metrics = {}
    ranks_np = np.asarray(ranks)
    for k in (10, 20):
        metrics[f"Recall@{k}"] = float(np.mean(ranks_np <= k))
        metrics[f"NDCG@{k}"] = float(np.mean([1 / math.log2(r + 1) if r <= k else 0 for r in ranks]))
        metrics[f"MRR@{k}"] = float(np.mean([1 / r if r <= k else 0 for r in ranks]))
    return metrics


def score_components(row, lambda_z, lambda_next, lambda_fine):
    return {
        "Anchor-only": z(row["anchor"]),
        "Past-only": z(row["stay"]),
        "Future-only": z(row["next"]),
        "Past+Future": z(row["stay"]) + lambda_next * z(row["next"]),
        "FIR-MIC without future bridge": z(row["anchor"]) + lambda_fine * z(row["fine_residual"]),
        "Per-item + transition/association": (
            z(row["anchor"]) + lambda_next * z(row["next"])
            + lambda_fine * z(row["fine_residual"])
        ),
        "FIR-MIC full": (
            z(row["anchor"]) + lambda_z * z(row["routed"])
            + lambda_next * z(row["next"]) + lambda_fine * z(row["fine_residual"])
        ),
    }


def evaluate(rows, lambda_z, lambda_next, lambda_fine, save_predictions=False):
    ranks, predictions = [], []
    for row_id, row in enumerate(rows):
        score = (z(row["anchor"]) + lambda_z * z(row["routed"]) +
                 lambda_next * z(row["next"]) + lambda_fine * z(row["fine_residual"]))
        pos = row["target_pos"]
        rank = stable_rank(score, pos)
        ranks.append(rank)
        if save_predictions:
            sample = row["sample"]
            predictions.append({
                "row_id": row_id, "user": sample.user, "target": sample.target,
                "rank": rank, "history_length": len(sample.prefix),
                "predicted_stay": row["route"], "target_stay_similarity": row["true_stay"],
                "target_score": float(score[pos]),
            })
    return metrics_from_ranks(ranks), predictions


def component_evidence(rows, locked):
    metric_ranks = defaultdict(list)
    predictions = []
    for row_id, row in enumerate(rows):
        scores = score_components(row, **locked)
        ranks = {name: stable_rank(score, row["target_pos"]) for name, score in scores.items()}
        for name, rank in ranks.items():
            metric_ranks[name].append(rank)
        past_cosine = float(np.dot(row["current"], row["target_q"]) /
                            max(np.linalg.norm(row["current"]) * np.linalg.norm(row["target_q"]), 1e-9))
        future_cosine = float(np.dot(row["future"], row["target_q"]) /
                              max(np.linalg.norm(row["future"]) * np.linalg.norm(row["target_q"]), 1e-9))
        sample = row["sample"]
        predictions.append({
            "row_id": row_id, "user": sample.user, "target": sample.target,
            "history_length": len(sample.prefix),
            "past_target_cosine": past_cosine,
            "future_target_cosine": future_cosine,
            "delta_future_minus_past": future_cosine - past_cosine,
            "future_better": int(future_cosine > past_cosine),
            **{f"rank_{name}": rank for name, rank in ranks.items()},
        })
    metrics = [{"variant": name, **metrics_from_ranks(ranks)} for name, ranks in metric_ranks.items()]
    return metrics, predictions


def route_diagnostics(rows):
    pred = np.asarray([r["route"] for r in rows])
    truth = np.asarray([r["true_stay"] for r in rows])
    hard = truth >= np.median(truth)
    if hard.all() or (~hard).all():
        auc = float("nan")
    else:
        order = np.argsort(pred)
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(pred) + 1)
        n1, n0 = hard.sum(), (~hard).sum()
        auc = float((ranks[hard].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return {
        "mean_predicted_stay": float(pred.mean()), "mean_target_stay_similarity": float(truth.mean()),
        "stay_spearman_proxy": float(np.corrcoef(pred.argsort().argsort(), truth.argsort().argsort())[0, 1]),
        "median_split_auc": auc,
    }


def teacher_for_modality(image, text, warm, k, seed, modality):
    if modality == "full":
        return warm_teacher(image, text, warm, k, seed)
    features = image if modality == "image" else text
    warm_ids = np.asarray(sorted(warm))
    _, centers, _, _ = intent_space(features[warm_ids], k, seed)
    logits = features @ centers.T / 0.12
    logits -= logits.max(1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(1, keepdims=True)
    counts = np.bincount(probabilities[warm_ids].argmax(1), minlength=k)
    return probabilities.astype(np.float32), {
        "fit_items": "warm-only", "intents": k, "modality": modality,
        "min_warm_cluster": int(counts.min()), "max_warm_cluster": int(counts.max()),
    }


@torch.no_grad()
def hard_item_intents(model, item_count, batch_size=4096):
    output = np.empty(item_count, dtype=np.int64)
    for start in range(0, item_count, batch_size):
        stop = min(item_count, start + batch_size)
        ids = torch.arange(start, stop, device=model.image.device)
        output[start:stop] = model.item_q(ids).argmax(-1).cpu().numpy()
    return output


def transition_statistics(model, train_samples, test_samples, item_count):
    states = hard_item_intents(model, item_count)
    counts = np.zeros((model.k, model.k), dtype=np.int64)
    for sample in train_samples:
        counts[states[sample.prefix[-1]], states[sample.target]] += 1
    out_degree = (counts > 0).sum(1)
    active = counts.sum(1) > 0
    entropies = []
    for row in counts[active]:
        probabilities = row[row > 0] / row.sum()
        entropies.append(float(-(probabilities * np.log(probabilities)).sum()))
    usable = []
    last_has_outgoing = []
    for sample in test_samples:
        sequence = list(sample.prefix)
        usable.append(sum(counts[states[a], states[b]] > 0 for a, b in zip(sequence, sequence[1:])))
        last_has_outgoing.append(out_degree[states[sequence[-1]]] > 0)
    edges = int((counts > 0).sum())
    return {
        "transition_nodes": int(model.k),
        "active_transition_nodes": int(active.sum()),
        "transition_edges": edges,
        "graph_density": edges / float(model.k * model.k),
        "average_out_degree": float(out_degree.mean()),
        "median_out_degree": float(np.median(out_degree)),
        "pct_states_without_outgoing": float(np.mean(out_degree == 0)),
        "mean_transition_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "pct_test_users_future_state_created": 1.0,
        "pct_test_users_last_state_has_observed_outgoing": float(np.mean(last_has_outgoing)),
        "average_usable_transitions_per_user": float(np.mean(usable)),
    }


def protocol_statistics(interactions, warm, validation_cold, test_cold, cutoff, train, validation, test):
    history = np.asarray([len(sample.prefix) for sample in test])
    warm_interactions = sum(item in warm and timestamp < cutoff for _, item, timestamp in interactions)
    return {
        "users": len({user for user, _, _ in interactions}),
        "warm_items": len(warm), "validation_cold_items": len(validation_cold),
        "test_cold_items": len(test_cold),
        "train_interactions": warm_interactions, "train_sequence_samples": len(train),
        "validation_interactions": len(validation),
        "test_interactions": len(test),
        "average_history_length": float(history.mean()),
        "median_history_length": float(np.median(history)),
        "pct_users_history_ge_2": float(np.mean(history >= 2)),
        "pct_users_history_ge_5": float(np.mean(history >= 5)),
        "cold_candidates_per_user": len(test_cold),
    }


def model_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def score_weight_grid(cache):
    rows = []
    values = (0.0, 0.05, 0.10, 0.20)
    for lambda_z in values:
        for lambda_next in values:
            for lambda_fine in values:
                metrics, _ = evaluate(cache, lambda_z, lambda_next, lambda_fine)
                rows.append({
                    "lambda_z": lambda_z, "lambda_next": lambda_next,
                    "lambda_fine": lambda_fine, **metrics,
                })
    best = max(rows, key=lambda row: (row["NDCG@10"], row["Recall@10"], row["MRR@10"]))
    return rows, best


def reset_random_state(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def grid_search(args, image_np, text_np, image, text, warm, pseudo_train,
                validation, validation_cold, device):
    """Select architecture, optimizer and epoch using validation only."""
    combinations = list(itertools.product(
        args.grid_intents, args.grid_dims, args.grid_learning_rates,
        args.grid_weight_decays, args.grid_dropouts, args.grid_router_dropouts,
    ))
    epoch_rows, trial_rows = [], []
    teacher_cache = {}
    checkpoint_dir = args.output_dir / "search_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    search_start = time.perf_counter()
    for trial_id, (intents, dim, learning_rate, weight_decay, dropout,
                   router_dropout) in enumerate(combinations, 1):
        fixed = {
            "trial": trial_id, "intents": intents, "dim": dim,
            "learning_rate": learning_rate, "weight_decay": weight_decay,
            "dropout": dropout, "router_dropout": router_dropout,
        }
        trial_checkpoint = checkpoint_dir / f"trial_{trial_id:04d}.json"
        if args.resume and trial_checkpoint.exists():
            try:
                saved = json.loads(trial_checkpoint.read_text(encoding="utf8"))
            except (json.JSONDecodeError, OSError):
                saved = {}
            if (saved.get("version") == GRID_SEARCH_VERSION
                    and saved.get("parameters") == fixed):
                epoch_rows.extend(saved["epochs"])
                trial_rows.append(saved["summary"])
                print(json.dumps({"resumed_completed_grid_trial": trial_id}), flush=True)
                continue

        reset_random_state(args.seed)
        if intents not in teacher_cache:
            teacher_cache[intents] = teacher_for_modality(
                image_np, text_np, warm, intents, args.seed, "full"
            )
        teacher, intent_diag = teacher_cache[intents]
        model = FIRMIC(
            image, text, torch.from_numpy(teacher).to(device), intents, dim, 10,
            modality="full", dropout=dropout, router_dropout=router_dropout,
        ).to(device)

        def validate_epoch(current_model, _epoch):
            cache = make_cache(
                current_model, validation, sorted(validation_cold), image, text,
                args.batch_size, device,
            )
            _, best = score_weight_grid(cache)
            del cache
            return best

        trial_start = time.perf_counter()
        history = train_model(
            model, pseudo_train, image, text, args.epochs, args.batch_size,
            args.seed, device, learning_rate=learning_rate,
            weight_decay=weight_decay, epoch_callback=validate_epoch,
            patience=args.grid_patience,
        )
        current_epoch_rows = []
        for row in history:
            current_epoch_rows.append({**fixed, **row})
        epoch_rows.extend(current_epoch_rows)
        best_epoch_row = max(
            history,
            key=lambda row: (
                row["validation_NDCG@10"], row["validation_Recall@10"],
                row["validation_MRR@10"], -row["epoch"],
            ),
        )
        trial_summary = {
            **fixed, "best_epoch": best_epoch_row["epoch"],
            "epochs_ran": len(history),
            "validation_NDCG@10": best_epoch_row["validation_NDCG@10"],
            "validation_Recall@10": best_epoch_row["validation_Recall@10"],
            "validation_MRR@10": best_epoch_row["validation_MRR@10"],
            "validation_Recall@20": best_epoch_row["validation_Recall@20"],
            "validation_NDCG@20": best_epoch_row["validation_NDCG@20"],
            "validation_MRR@20": best_epoch_row["validation_MRR@20"],
            "lambda_z": best_epoch_row["validation_lambda_z"],
            "lambda_next": best_epoch_row["validation_lambda_next"],
            "lambda_fine": best_epoch_row["validation_lambda_fine"],
            "trial_seconds": time.perf_counter() - trial_start,
            "intent_initialization": json.dumps(intent_diag, sort_keys=True),
        }
        trial_rows.append(trial_summary)
        trial_checkpoint.write_text(json.dumps({
            "version": GRID_SEARCH_VERSION,
            "parameters": fixed, "epochs": current_epoch_rows,
            "summary": trial_summary,
        }, indent=2), encoding="utf8")
        write_csv(args.output_dir / "hyperparameter_grid.csv", epoch_rows)
        write_csv(args.output_dir / "hyperparameter_trials.csv", trial_rows)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    best = max(
        trial_rows,
        key=lambda row: (
            row["validation_NDCG@10"], row["validation_Recall@10"],
            row["validation_MRR@10"], -row["best_epoch"],
        ),
    )
    optimal = {
        key: best[key] for key in (
            "trial", "intents", "dim", "learning_rate", "weight_decay",
            "dropout", "router_dropout", "best_epoch",
        )
    }
    optimal["selection_split"] = "validation only"
    optimal["grid_search_version"] = GRID_SEARCH_VERSION
    optimal["selection_metric"] = "lexicographic NDCG@10, Recall@10, MRR@10; earlier epoch tie-break"
    optimal["search_max_epochs"] = args.epochs
    optimal["grid_patience"] = args.grid_patience
    optimal["search_seconds"] = sum(row["trial_seconds"] for row in trial_rows)
    optimal["resume_overhead_seconds"] = time.perf_counter() - search_start
    optimal["selected_validation"] = {
        key[len("validation_"):]: best[key]
        for key in (
            "validation_Recall@10", "validation_NDCG@10", "validation_MRR@10",
            "validation_Recall@20", "validation_NDCG@20", "validation_MRR@20",
        )
    }
    optimal["search_space"] = {
        "intents": args.grid_intents, "dims": args.grid_dims,
        "learning_rates": args.grid_learning_rates,
        "weight_decays": args.grid_weight_decays,
        "dropouts": args.grid_dropouts,
        "router_dropouts": args.grid_router_dropouts,
    }
    optimal["completed_trials"] = len(trial_rows)
    optimal["total_trials"] = len(combinations)
    return optimal, epoch_rows, trial_rows


def run_modality_ablation(modality, args, image_np, text_np, warm, pseudo_train,
                          validation, validation_cold, test, test_cold, device,
                          optimal):
    reset_random_state(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed); torch.cuda.reset_peak_memory_stats()
    teacher, _ = teacher_for_modality(
        image_np, text_np, warm, optimal["intents"], args.seed, modality
    )
    image = torch.from_numpy(image_np).to(device); text = torch.from_numpy(text_np).to(device)
    model = FIRMIC(
        image, text, torch.from_numpy(teacher).to(device), optimal["intents"],
        optimal["dim"], 10, modality=modality, dropout=optimal["dropout"],
        router_dropout=optimal["router_dropout"],
    ).to(device)
    train_start = time.perf_counter()
    training = train_model(
        model, pseudo_train, image, text, optimal["best_epoch"],
        args.batch_size, args.seed, device,
        learning_rate=optimal["learning_rate"], weight_decay=optimal["weight_decay"],
    )
    train_seconds = time.perf_counter() - train_start
    validation_cache = make_cache(model, validation, sorted(validation_cold), image, text,
                                  args.batch_size, device)
    grid, best = score_weight_grid(validation_cache)
    locked = {key: best[key] for key in ("lambda_z", "lambda_next", "lambda_fine")}
    inference_start = time.perf_counter()
    test_cache = make_cache(model, test, sorted(test_cold), image, text, args.batch_size, device)
    inference_seconds = time.perf_counter() - inference_start
    scores, predictions = evaluate(test_cache, **locked, save_predictions=True)
    prefix = f"{modality}_only"
    write_csv(args.output_dir / f"{prefix}_training.csv", training)
    write_csv(args.output_dir / f"{prefix}_validation_grid.csv", grid)
    write_csv(args.output_dir / f"{prefix}_predictions.csv", predictions)
    (args.output_dir / f"{prefix}_locked_config.json").write_text(
        json.dumps(locked, indent=2), encoding="utf8"
    )
    torch.save({"state_dict": model.state_dict(), "locked": locked},
               args.output_dir / f"{prefix}_model.pt")
    row = {
        "variant": f"FIR-MIC {modality}-only", **scores,
        "train_seconds": train_seconds,
        "inference_ms_per_user": 1000 * inference_seconds / len(test),
        "trainable_parameters": model_parameter_count(model),
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        ),
    }
    del model, image, text, validation_cache, test_cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return row


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--dataset", choices=("baby", "clothing", "sports"), default="baby")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Maximum epochs evaluated for every grid trial")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grid-intents", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--grid-dims", nargs="+", type=int, default=[64])
    parser.add_argument("--grid-learning-rates", nargs="+", type=float,
                        default=[5e-4, 1e-3])
    parser.add_argument("--grid-weight-decays", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--grid-dropouts", nargs="+", type=float, default=[0.15])
    parser.add_argument("--grid-router-dropouts", nargs="+", type=float, default=[0.10])
    parser.add_argument("--grid-patience", type=int, default=5,
                        help="Stop a trial after this many validation epochs without improvement; 0 disables")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed grid-trial checkpoints in the output directory")
    parser.add_argument("--validation-users", type=int)
    parser.add_argument("--test-users", type=int)
    parser.add_argument("--extended-ablations", action="store_true",
                        help="Retrain image-only and text-only FIR-MIC variants")
    parser.add_argument("--data-dir", type=Path, default=ROOT,
                        help="MMREC-COLD root containing baby/baby.inter and feature arrays")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/baby_fir_mic_seed2022")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.grid_patience < 0:
        parser.error("--grid-patience cannot be negative")
    if any(dim <= 0 or dim % 4 for dim in args.grid_dims):
        parser.error("every --grid-dims value must be positive and divisible by 4")
    if any(value <= 0 for value in args.grid_intents + args.grid_learning_rates):
        parser.error("intent counts and learning rates must be positive")
    if any(value < 0 for value in args.grid_weight_decays):
        parser.error("weight decays cannot be negative")
    if any(not 0 <= value < 1 for value in args.grid_dropouts + args.grid_router_dropouts):
        parser.error("dropout values must be in [0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    data_root = args.data_dir / args.dataset
    interactions = read_interactions(data_root, args.dataset)
    warm, validation_cold, test_cold, cutoff, _ = temporal_item_split(interactions, Config(max_sequence_length=10))
    train, validation, test, _ = build_samples(
        interactions, warm, validation_cold, test_cold, cutoff, 10
    )
    checks = leakage_checks(train, validation, test, warm, validation_cold, test_cold)
    pseudo_train = pseudo_cold(train)
    if max(args.grid_intents) > len(warm):
        parser.error("an intent count cannot exceed the number of warm items")
    if args.validation_users: validation = validation[:args.validation_users]
    if args.test_users: test = test[:args.test_users]

    start = time.perf_counter()
    image_np = l2_blocks(np.load(data_root / "image_feat.npy", mmap_mode="r"))
    text_np = l2_blocks(np.load(data_root / "text_feat.npy", mmap_mode="r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    image = torch.from_numpy(image_np).to(device); text = torch.from_numpy(text_np).to(device)
    preprocessing_seconds = time.perf_counter() - start

    optimal, search_epochs, search_trials = grid_search(
        args, image_np, text_np, image, text, warm, pseudo_train,
        validation, validation_cold, device,
    )
    write_csv(args.output_dir / "hyperparameter_grid.csv", search_epochs)
    write_csv(args.output_dir / "hyperparameter_trials.csv", search_trials)
    (args.output_dir / "optimal_config.json").write_text(
        json.dumps(optimal, indent=2), encoding="utf8"
    )

    # Confirmatory fit: start from a fresh initialization and train for the
    # validation-selected epoch count. Test data have not been read by scoring
    # or model selection before this point.
    reset_random_state(args.seed)
    teacher, intent_diag = teacher_for_modality(
        image_np, text_np, warm, optimal["intents"], args.seed, "full"
    )
    model = FIRMIC(
        image, text, torch.from_numpy(teacher).to(device), optimal["intents"],
        optimal["dim"], 10, modality="full", dropout=optimal["dropout"],
        router_dropout=optimal["router_dropout"],
    ).to(device)
    train_start = time.perf_counter()
    training = train_model(
        model, pseudo_train, image, text, optimal["best_epoch"],
        args.batch_size, args.seed, device,
        learning_rate=optimal["learning_rate"], weight_decay=optimal["weight_decay"],
    )
    train_seconds = time.perf_counter() - train_start
    write_csv(args.output_dir / "training.csv", training)

    validation_cache = make_cache(model, validation, sorted(validation_cold), image, text, args.batch_size, device)
    grid, best = score_weight_grid(validation_cache)
    locked = {key: best[key] for key in ("lambda_z", "lambda_next", "lambda_fine")}
    optimal["locked_score_weights"] = locked
    optimal["confirmatory_validation"] = {
        key: best[key] for key in (
            "Recall@10", "NDCG@10", "MRR@10", "Recall@20", "NDCG@20", "MRR@20"
        )
    }
    (args.output_dir / "optimal_config.json").write_text(
        json.dumps(optimal, indent=2), encoding="utf8"
    )
    write_csv(args.output_dir / "validation_grid.csv", grid)
    (args.output_dir / "locked_config.json").write_text(json.dumps(locked, indent=2), encoding="utf8")

    inference_start = time.perf_counter()
    test_cache = make_cache(model, test, sorted(test_cold), image, text, args.batch_size, device)
    inference_seconds = time.perf_counter() - inference_start
    metrics, predictions = evaluate(test_cache, **locked, save_predictions=True)
    write_csv(args.output_dir / "predictions.csv", predictions)
    component_metrics, component_predictions = component_evidence(test_cache, locked)
    write_csv(args.output_dir / "component_ablation.csv", component_metrics)
    write_csv(args.output_dir / "component_predictions.csv", component_predictions)
    full_peak_cuda_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    )

    modality_rows = [{
        "variant": "FIR-MIC full", **metrics,
        "train_seconds": train_seconds,
        "inference_ms_per_user": 1000 * inference_seconds / len(test),
        "trainable_parameters": model_parameter_count(model),
        "peak_cuda_memory_mb": full_peak_cuda_mb,
    }]
    if args.extended_ablations:
        for modality in ("image", "text"):
            modality_rows.append(run_modality_ablation(
                modality, args, image_np, text_np, warm, pseudo_train, validation,
                validation_cold, test, test_cold, device, optimal
            ))
    write_csv(args.output_dir / "modality_ablation.csv", modality_rows)

    transition_stats = transition_statistics(model, train, test, len(image_np))
    protocol_stats = protocol_statistics(
        interactions, warm, validation_cold, test_cold, cutoff, train, validation, test
    )
    efficiency = {
        "trainable_parameters": model_parameter_count(model),
        "hyperparameter_search_seconds": optimal["search_seconds"],
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_user": 1000 * inference_seconds / len(test),
        "peak_cuda_memory_mb": full_peak_cuda_mb,
    }
    (args.output_dir / "transition_statistics.json").write_text(
        json.dumps(transition_stats, indent=2), encoding="utf8"
    )
    (args.output_dir / "protocol_statistics.json").write_text(
        json.dumps(protocol_stats, indent=2), encoding="utf8"
    )
    (args.output_dir / "efficiency.json").write_text(
        json.dumps(efficiency, indent=2), encoding="utf8"
    )

    result = {
        "method": "FIR-MIC", "dataset": args.dataset, "seed": args.seed, "device": str(device),
        "protocol": "strict temporal item-cold full ranking; warm-only pseudo-cold training",
        "features": "original image4096 and text384 consumed by separate full-input adapters",
        "intents": optimal["intents"], "dim": optimal["dim"],
        "epochs": optimal["best_epoch"], "search_max_epochs": args.epochs,
        "optimal_config": optimal,
        "train_samples_original": len(train), "train_samples_pseudo_cold": len(pseudo_train),
        "validation_users": len(validation), "test_users": len(test),
        "validation_locked": locked, "test": metrics,
        "route_diagnostics": route_diagnostics(test_cache),
        "transition_statistics": transition_stats,
        "protocol_statistics": protocol_stats,
        "efficiency": efficiency,
        "intent_initialization": intent_diag, "leakage_checks": checks,
        "preprocessing_seconds": preprocessing_seconds, "train_seconds": train_seconds,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    torch.save(
        {"state_dict": model.state_dict(), "optimal_config": optimal, "result": result},
        args.output_dir / "model.pt",
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
