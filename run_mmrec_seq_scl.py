"""Run static and sequential score-level contrastive models on MMREC datasets.

The supplied x_label is an interaction split and is not item-cold. This runner
constructs a strict temporal item split from each item's first timestamp:
earliest 80% warm, next 10% validation-cold, latest 10% test-cold.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


TOPK = (1, 5, 10, 20)
MODEL_SPECS = {
    "static": ("Static score-level", "static", 0.0),
    "seq-scl": ("Seq-SCL InfoNCE", "sequential", 0.0),
    "seq-scl-top1": ("Seq-SCL InfoNCE + top1", "sequential", "top1"),
    "hybrid": ("Hybrid Mean-GRU", "hybrid", 0.0),
    "multi-item": ("Multi-interest per-item", "multi_item", 0.0),
    "multi-cluster": ("Multi-interest clustered", "multi_cluster", 0.0),
}


@dataclass
class Config:
    seeds: tuple[int, ...] = (2022, 2023, 2025)
    warm_ratio: float = 0.8
    validation_ratio: float = 0.1
    alpha: float = 0.2
    hidden_dim: int = 128
    image_reduced_dim: int = 256
    dropout: float = 0.2
    tau: float = 0.07
    margin: float = 0.2
    lambda_top1: float = 0.5
    epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 128
    max_sequence_length: int = 10
    interest_clusters: int = 3
    interest_aggregation_tau: float = 0.10
    cluster_iterations: int = 3
    validation_users_for_checkpoint: int = 2000
    device: str = "auto"


@dataclass(frozen=True)
class Sample:
    user: int
    prefix: tuple[int, ...]
    target: int
    prefix_times: tuple[int, ...]
    target_time: int


class SequenceDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(rows: list[Sample]):
    lengths = torch.tensor([len(row.prefix) for row in rows], dtype=torch.long)
    width = int(lengths.max())
    prefixes = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        prefixes[index, : len(row.prefix)] = torch.tensor(row.prefix)
    return {
        "users": [row.user for row in rows],
        "prefixes": prefixes,
        "lengths": lengths,
        "targets": torch.tensor([row.target for row in rows], dtype=torch.long),
        "target_times": [row.target_time for row in rows],
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_interactions(root: Path, name: str):
    interactions = []
    with (root / f"{name}.inter").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            interactions.append(
                (int(row["userID"]), int(row["itemID"]), int(row["timestamp"]))
            )
    return interactions


def temporal_item_split(interactions: list[tuple[int, int, int]], config: Config):
    first_timestamp = {}
    for _, item, timestamp in interactions:
        first_timestamp[item] = min(first_timestamp.get(item, timestamp), timestamp)
    ordered = sorted(first_timestamp, key=lambda item: (first_timestamp[item], item))
    warm_end = int(len(ordered) * config.warm_ratio)
    validation_end = int(
        len(ordered) * (config.warm_ratio + config.validation_ratio)
    )
    warm = set(ordered[:warm_end])
    validation = set(ordered[warm_end:validation_end])
    test = set(ordered[validation_end:])
    validation_start = min(first_timestamp[item] for item in validation)
    test_start = min(first_timestamp[item] for item in test)
    return warm, validation, test, validation_start, test_start


def build_samples(
    interactions: list[tuple[int, int, int]],
    warm: set[int],
    validation: set[int],
    test: set[int],
    validation_start: int,
    max_length: int,
):
    by_user = defaultdict(list)
    for user, item, timestamp in interactions:
        by_user[user].append((timestamp, item))
    train_samples, validation_samples, test_samples = [], [], []
    prefix_lengths = {"validation": Counter(), "test": Counter()}
    for user, history in by_user.items():
        history.sort(key=lambda pair: (pair[0], pair[1]))
        train_history = [
            pair
            for pair in history
            if pair[1] in warm and pair[0] < validation_start
        ]
        for position in range(1, len(train_history)):
            target_time, target_item = train_history[position]
            # Several reviews can share a day-level timestamp. Never use an
            # arbitrary item-ID tie-break as if it were temporal evidence.
            prefix_rows = [
                pair for pair in train_history[:position] if pair[0] < target_time
            ][-max_length:]
            if not prefix_rows:
                continue
            train_samples.append(
                Sample(
                    user=user,
                    prefix=tuple(item for _, item in prefix_rows),
                    target=target_item,
                    prefix_times=tuple(timestamp for timestamp, _ in prefix_rows),
                    target_time=target_time,
                )
            )
        for split_name, cold_items, output in (
            ("validation", validation, validation_samples),
            ("test", test, test_samples),
        ):
            targets = [pair for pair in history if pair[1] in cold_items]
            if not targets:
                continue
            target_time, target = targets[0]
            prefix_rows = [
                pair
                for pair in history
                if pair[1] in warm and pair[0] < target_time
            ][-max_length:]
            if not prefix_rows:
                continue
            output.append(
                Sample(
                    user=user,
                    prefix=tuple(item for _, item in prefix_rows),
                    target=target,
                    prefix_times=tuple(timestamp for timestamp, _ in prefix_rows),
                    target_time=target_time,
                )
            )
            prefix_lengths[split_name][len(prefix_rows)] += 1
    return train_samples, validation_samples, test_samples, prefix_lengths


def leakage_checks(
    train_samples: list[Sample],
    validation_samples: list[Sample],
    test_samples: list[Sample],
    warm: set[int],
    validation: set[int],
    test: set[int],
):
    assert warm.isdisjoint(validation)
    assert warm.isdisjoint(test)
    assert validation.isdisjoint(test)
    for sample in train_samples:
        assert sample.target in warm
        assert set(sample.prefix).issubset(warm)
        assert all(value < sample.target_time for value in sample.prefix_times)
    for samples, targets in (
        (validation_samples, validation),
        (test_samples, test),
    ):
        for sample in samples:
            assert sample.target in targets
            assert set(sample.prefix).issubset(warm)
            assert all(value < sample.target_time for value in sample.prefix_times)
    return {
        "item_sets_disjoint": True,
        "timestamp_order_valid": True,
        "cold_item_leakage": False,
        "train_samples_checked": len(train_samples),
        "validation_samples_checked": len(validation_samples),
        "test_samples_checked": len(test_samples),
    }


def load_features(root: Path, config: Config):
    image = np.load(root / "image_feat.npy", mmap_mode="r")
    text = np.load(root / "text_feat.npy", mmap_mode="r")
    if image.shape[1] % config.image_reduced_dim:
        raise ValueError("Image dimension is not divisible by requested reduced dim")
    block = image.shape[1] // config.image_reduced_dim
    # The same deterministic reduction is applied to every item and has no
    # item-specific parameters.
    image_reduced = np.asarray(image, dtype=np.float32).reshape(
        image.shape[0], config.image_reduced_dim, block
    ).mean(axis=2)
    image_tensor = F.normalize(torch.from_numpy(image_reduced), dim=1)
    text_tensor = F.normalize(
        torch.from_numpy(np.asarray(text, dtype=np.float32).copy()), dim=1
    )
    return image_tensor, text_tensor, tuple(image.shape), tuple(text.shape)


class ScoreModel(nn.Module):
    def __init__(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        config: Config,
        architecture: str,
    ):
        super().__init__()
        self.register_buffer("image_features", image_features)
        self.register_buffer("text_features", text_features)
        self.image_projection = nn.Linear(image_features.shape[1], config.hidden_dim)
        self.text_projection = nn.Linear(text_features.shape[1], config.hidden_dim)
        if architecture not in {
            "static",
            "sequential",
            "hybrid",
            "multi_item",
            "multi_cluster",
        }:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.architecture = architecture
        self.sequential = architecture in {"sequential", "hybrid"}
        self.alpha = config.alpha
        self.interest_clusters = config.interest_clusters
        self.interest_aggregation_tau = config.interest_aggregation_tau
        self.cluster_iterations = config.cluster_iterations
        self.dropout = nn.Dropout(config.dropout)
        if self.sequential:
            self.image_gru = nn.GRU(
                config.hidden_dim, config.hidden_dim, batch_first=True
            )
            self.text_gru = nn.GRU(
                config.hidden_dim, config.hidden_dim, batch_first=True
            )
        if architecture == "hybrid":
            # The gate starts near mean pooling for short histories and grows
            # with log prefix length. It remains fully trainable.
            self.length_gate = nn.Linear(1, 1)
            nn.init.constant_(self.length_gate.weight, 1.5)
            nn.init.constant_(self.length_gate.bias, -3.0)

    def project(self, image: torch.Tensor, text: torch.Tensor):
        return F.normalize(self.image_projection(image), dim=-1), F.normalize(
            self.text_projection(text), dim=-1
        )

    def encode(self, prefixes: torch.Tensor, lengths: torch.Tensor):
        image, text = self.project(
            self.image_features[prefixes], self.text_features[prefixes]
        )
        positions = torch.arange(
            prefixes.shape[1], device=prefixes.device
        )[None, :]
        mask = (positions < lengths[:, None]).float().unsqueeze(-1)
        image_mean = F.normalize(
            (image * mask).sum(1) / lengths[:, None], dim=-1
        )
        text_mean = F.normalize(
            (text * mask).sum(1) / lengths[:, None], dim=-1
        )
        if self.architecture == "static":
            return image_mean, text_mean

        if self.architecture == "multi_item":
            # Every valid warm item is an interest. Invalid padded positions
            # remain exactly zero so score_candidates can mask them.
            return image * mask, text * mask

        if self.architecture == "multi_cluster":
            return self.cluster_interests(image, text, lengths)

        image_sequence = self.dropout(image)
        text_sequence = self.dropout(text)
        packed_image = pack_padded_sequence(
            image_sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_text = pack_padded_sequence(
            text_sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, image_hidden = self.image_gru(packed_image)
        _, text_hidden = self.text_gru(packed_text)
        image_gru = F.normalize(image_hidden[-1], dim=-1)
        text_gru = F.normalize(text_hidden[-1], dim=-1)
        if self.architecture == "sequential":
            return image_gru, text_gru

        gate = self.gate_values(lengths)
        image_state = F.normalize(
            (1.0 - gate) * image_mean + gate * image_gru, dim=-1
        )
        text_state = F.normalize(
            (1.0 - gate) * text_mean + gate * text_gru, dim=-1
        )
        return image_state, text_state

    def cluster_interests(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        lengths: torch.Tensor,
    ):
        """Deterministic cosine k-means with differentiable cluster means.

        Assignments are computed without gradients. The projected image/text
        vectors inside each selected cluster are then averaged with gradients,
        so both modality projections still learn from score-level InfoNCE.
        """
        batch_size, width, _ = image.shape
        max_clusters = min(self.interest_clusters, width)
        combined = F.normalize(
            self.alpha * image + (1.0 - self.alpha) * text, dim=-1
        )
        positions = torch.arange(width, device=image.device)[None, :]
        valid_items = positions < lengths[:, None]
        cluster_ids = torch.arange(max_clusters, device=image.device)[None, :]
        cluster_counts = lengths.clamp_max(max_clusters)
        valid_clusters = cluster_ids < cluster_counts[:, None]

        # Evenly spaced prefix positions provide deterministic, distinct
        # initial centers whenever length >= K; batched k-means then groups by
        # multimodal cosine similarity without a Python loop over users.
        initial_positions = torch.floor(
            (cluster_ids.float() + 0.5)
            * lengths[:, None].float()
            / cluster_counts[:, None].float()
        ).long()
        initial_positions = initial_positions.clamp_max(width - 1)
        gather_index = initial_positions.unsqueeze(-1).expand(
            -1, -1, combined.shape[-1]
        )
        centers = combined.detach().gather(1, gather_index)

        assignments = None
        for _ in range(self.cluster_iterations):
            similarities = torch.einsum(
                "bld,bkd->blk", combined.detach(), centers
            )
            similarities = similarities.masked_fill(
                ~valid_items.unsqueeze(-1), -torch.inf
            ).masked_fill(~valid_clusters.unsqueeze(1), -torch.inf)
            assignments = similarities.argmax(dim=-1)
            membership = F.one_hot(
                assignments, num_classes=max_clusters
            ).to(combined.dtype)
            membership = membership * valid_items.unsqueeze(-1)
            member_counts = membership.sum(dim=1)
            updated = torch.einsum(
                "blk,bld->bkd", membership, combined.detach()
            ) / member_counts.clamp_min(1).unsqueeze(-1)
            updated = F.normalize(updated, dim=-1)
            centers = torch.where(
                (member_counts > 0).unsqueeze(-1), updated, centers
            )

        membership = F.one_hot(
            assignments, num_classes=max_clusters
        ).to(image.dtype)
        membership = membership * valid_items.unsqueeze(-1)
        member_counts = membership.sum(dim=1)
        denominator = member_counts.clamp_min(1).unsqueeze(-1)
        image_interests = F.normalize(
            torch.einsum("blk,bld->bkd", membership, image) / denominator,
            dim=-1,
        )
        text_interests = F.normalize(
            torch.einsum("blk,bld->bkd", membership, text) / denominator,
            dim=-1,
        )
        populated = valid_clusters & (member_counts > 0)
        return (
            image_interests * populated.unsqueeze(-1),
            text_interests * populated.unsqueeze(-1),
        )

    def gate_values(self, lengths: torch.Tensor):
        if self.architecture != "hybrid":
            raise RuntimeError("Length gate is only available for hybrid models")
        length_feature = torch.log1p(lengths.float()).unsqueeze(-1)
        return torch.sigmoid(self.length_gate(length_feature))

    def score_candidates(
        self,
        image_state: torch.Tensor,
        text_state: torch.Tensor,
        candidate_indices: torch.Tensor,
    ):
        image, text = self.project(
            self.image_features[candidate_indices],
            self.text_features[candidate_indices],
        )
        if image_state.ndim == 2:
            return self.alpha * (image_state @ image.T) + (1.0 - self.alpha) * (
                text_state @ text.T
            )

        image_scores = torch.einsum("bkd,cd->bkc", image_state, image)
        text_scores = torch.einsum("bkd,cd->bkc", text_state, text)
        interest_scores = (
            self.alpha * image_scores + (1.0 - self.alpha) * text_scores
        )
        valid = image_state.square().sum(dim=-1) > 0
        scaled = (interest_scores / self.interest_aggregation_tau).masked_fill(
            ~valid.unsqueeze(-1), -torch.inf
        )
        # log-mean-exp is invariant to the number of interests for ranking and
        # prevents longer histories from receiving an artificial score bonus.
        normalizer = valid.sum(dim=1).clamp_min(1).log().unsqueeze(-1)
        return self.interest_aggregation_tau * (
            torch.logsumexp(scaled, dim=1) - normalizer
        )


class MultiPositiveBatchLoss(nn.Module):
    def __init__(self, tau: float, margin: float, lambda_top1: float):
        super().__init__()
        self.tau = tau
        self.margin = margin
        self.lambda_top1 = lambda_top1

    def forward(self, scores: torch.Tensor, target_items: torch.Tensor):
        positive_mask = target_items[:, None] == target_items[None, :]
        scaled = scores / self.tau
        numerator = torch.logsumexp(
            scaled.masked_fill(~positive_mask, -torch.inf), dim=1
        )
        denominator = torch.logsumexp(scaled, dim=1)
        loss = denominator - numerator
        if self.lambda_top1 > 0:
            negative = scores.masked_fill(positive_mask, -torch.inf).max(dim=1).values
            positive = scores.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
            valid = torch.isfinite(negative)
            margin_loss = F.relu(self.margin + negative[valid] - positive[valid])
            if len(margin_loss):
                loss = loss.mean() + self.lambda_top1 * margin_loss.mean()
                return loss
        return loss.mean()


def ranking_metrics(ranks: list[int], recommendations: list[list[int]], count: int):
    output = {"n_users_eval": len(ranks), "n_candidates": count}
    for k in TOPK:
        output[f"Recall@{k}"] = statistics.fmean(rank <= k for rank in ranks)
        output[f"NDCG@{k}"] = statistics.fmean(
            1.0 / math.log2(rank + 1.0) if rank <= k else 0.0 for rank in ranks
        )
        output[f"MRR@{k}"] = statistics.fmean(
            1.0 / rank if rank <= k else 0.0 for rank in ranks
        )
    output["Coverage@20"] = (
        len({item for row in recommendations for item in row}) / count
    )
    return output


@torch.no_grad()
def evaluate(
    model: ScoreModel,
    samples: list[Sample],
    candidates: list[int],
    batch_size: int = 256,
    save_predictions: bool = False,
):
    model.eval()
    device = model.image_features.device
    candidate_tensor = torch.tensor(
        candidates, dtype=torch.long, device=device
    )
    candidate_lookup = {item: index for index, item in enumerate(candidates)}
    ranks, recommendations, predictions = [], [], []
    loader = DataLoader(
        SequenceDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    for batch in loader:
        prefixes = batch["prefixes"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        image_state, text_state = model.encode(prefixes, lengths)
        gate_values = (
            model.gate_values(lengths).squeeze(-1)
            if model.architecture == "hybrid"
            else None
        )
        scores = model.score_candidates(image_state, text_state, candidate_tensor)
        order = torch.argsort(scores, dim=1, descending=True)
        for row_index, (user, target) in enumerate(
            zip(batch["users"], batch["targets"].tolist())
        ):
            target_position = candidate_lookup[target]
            rank = int((order[row_index] == target_position).nonzero()[0]) + 1
            ranks.append(rank)
            recommendations.append(
                [candidates[index] for index in order[row_index, :20].tolist()]
            )
            if save_predictions:
                prediction = {
                    "user_id": user,
                    "target_item": target,
                    "prefix_length": int(lengths[row_index]),
                    "rank": rank,
                    "target_score": float(scores[row_index, target_position]),
                }
                if model.architecture == "multi_item":
                    prediction["n_interests"] = int(lengths[row_index])
                elif model.architecture == "multi_cluster":
                    prediction["n_interests"] = min(
                        model.interest_clusters, int(lengths[row_index])
                    )
                if gate_values is not None:
                    prediction["hybrid_gate"] = float(gate_values[row_index])
                predictions.append(prediction)
    return ranking_metrics(ranks, recommendations, len(candidates)), predictions


def train_model(
    seed: int,
    model_name: str,
    architecture: str,
    lambda_top1: float,
    train_samples: list[Sample],
    validation_samples: list[Sample],
    validation_items: list[int],
    image: torch.Tensor,
    text: torch.Tensor,
    config: Config,
    checkpoint_path: Path,
    resume: bool,
    on_epoch,
):
    set_seed(seed)
    if config.device == "auto":
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda") + 1
                device = torch.device("cuda")
            except Exception as e:
                print(f"[Device] CUDA device available but incompatible ({e}). Falling back to CPU.", flush=True)
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = ScoreModel(image, text, config, architecture).to(device)
    criterion = MultiPositiveBatchLoss(
        config.tau, config.margin, lambda_top1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        SequenceDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    validation_subset = validation_samples[: config.validation_users_for_checkpoint]
    best_state, best_epoch, best_ndcg = None, None, -1.0
    epoch_rows, gradient_check = [], None
    completed_epoch, previous_seconds = 0, 0.0
    if resume and checkpoint_path.exists():
        checkpoint = torch_load_checkpoint(checkpoint_path, device)
        previous_config = checkpoint["config"]
        current_config = asdict(config)
        ignored = {"epochs", "device", "seeds"}
        if architecture not in {"multi_item", "multi_cluster"}:
            ignored.update(
                {
                    "interest_clusters",
                    "interest_aggregation_tau",
                    "cluster_iterations",
                }
            )
        incompatible = {
            key: (previous_config.get(key), value)
            for key, value in current_config.items()
            if key not in ignored and previous_config.get(key) != value
        }
        if incompatible:
            raise ValueError(
                f"Checkpoint configuration mismatch: {incompatible}"
            )
        load_trainable_state(model, checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_state = checkpoint["best_state"]
        best_epoch = checkpoint["best_epoch"]
        best_ndcg = checkpoint["best_ndcg"]
        epoch_rows = checkpoint["epoch_rows"]
        gradient_check = checkpoint["gradient_check"]
        completed_epoch = checkpoint["completed_epoch"]
        previous_seconds = checkpoint.get("train_seconds", 0.0)
        generator.set_state(checkpoint["generator_state"].cpu())
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state"]]
            )
        print(
            f"RESUME {model_name} seed={seed} from epoch {completed_epoch}",
            flush=True,
        )
    start = time.perf_counter() - previous_seconds
    for epoch in range(completed_epoch + 1, config.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prefixes = batch["prefixes"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            image_state, text_state = model.encode(
                prefixes, lengths
            )
            scores = model.score_candidates(
                image_state, text_state, targets
            )
            loss = criterion(scores, targets)
            loss.backward()
            if gradient_check is None:
                gradient_check = {
                    "features_frozen": not (
                        model.image_features.requires_grad
                        or model.text_features.requires_grad
                    ),
                    "image_projection_grad": float(
                        model.image_projection.weight.grad.abs().sum()
                    ),
                    "text_projection_grad": float(
                        model.text_projection.weight.grad.abs().sum()
                    ),
                    "image_gru_grad": (
                        float(
                            sum(
                                parameter.grad.abs().sum()
                                for parameter in model.image_gru.parameters()
                                if parameter.grad is not None
                            )
                        )
                        if model.sequential
                        else None
                    ),
                    "text_gru_grad": (
                        float(
                            sum(
                                parameter.grad.abs().sum()
                                for parameter in model.text_gru.parameters()
                                if parameter.grad is not None
                            )
                        )
                        if model.sequential
                        else None
                    ),
                    "length_gate_grad": (
                        float(model.length_gate.weight.grad.abs().sum())
                        if architecture == "hybrid"
                        else None
                    ),
                }
            optimizer.step()
            losses.append(float(loss.detach()))
        validation_metrics, _ = evaluate(
            model, validation_subset, validation_items
        )
        epoch_loss = statistics.fmean(losses)
        epoch_rows.append(
            {
                "model": model_name,
                "seed": seed,
                "epoch": epoch,
                "loss": epoch_loss,
                "validation_NDCG@20": validation_metrics["NDCG@20"],
            }
        )
        if validation_metrics["NDCG@20"] > best_ndcg:
            best_ndcg = validation_metrics["NDCG@20"]
            best_epoch = epoch
            best_state = trainable_state(model)
        elapsed = time.perf_counter() - start
        atomic_torch_save(
            {
                "format_version": 1,
                "model_name": model_name,
                "seed": seed,
                "config": asdict(config),
                "completed_epoch": epoch,
                "model_state": trainable_state(model),
                "optimizer_state": optimizer.state_dict(),
                "best_state": best_state,
                "best_epoch": best_epoch,
                "best_ndcg": best_ndcg,
                "epoch_rows": epoch_rows,
                "gradient_check": gradient_check,
                "train_seconds": elapsed,
                "generator_state": generator.get_state(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            checkpoint_path,
        )
        print(
            f"CHECKPOINT {model_name} seed={seed} epoch={epoch} "
            f"path={checkpoint_path}",
            flush=True,
        )
        on_epoch(epoch_rows[-1], checkpoint_path)
    load_trainable_state(model, best_state)
    return (
        model,
        epoch_rows,
        {
            "best_epoch": best_epoch,
            "best_validation_NDCG@20": best_ndcg,
            "train_seconds": time.perf_counter() - start,
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
            "gradient_check": gradient_check,
            "loss_decreased": epoch_rows[-1]["loss"] < epoch_rows[0]["loss"],
        },
    )


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def upsert_csv(path: Path, new_rows: list[dict], key_fields: tuple[str, ...]):
    rows = read_csv(path)
    by_key = {
        tuple(str(row[field]) for field in key_fields): row for row in rows
    }
    for row in new_rows:
        key = tuple(str(row[field]) for field in key_fields)
        by_key[key] = row
    merged = list(by_key.values())
    write_csv(path, merged)
    return merged


def trainable_state(model: nn.Module):
    parameter_names = set(dict(model.named_parameters()))
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name in parameter_names
    }


def load_trainable_state(model: nn.Module, state: dict):
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"image_features", "text_features"}
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            f"Invalid trainable checkpoint: missing={missing}, "
            f"unexpected={unexpected}"
        )


def atomic_torch_save(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def torch_load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def git_environment():
    environment = os.environ.copy()
    token = environment.get("GITHUB_TOKEN", "").strip()
    if token:
        encoded = base64.b64encode(
            f"x-access-token:{token}".encode("utf-8")
        ).decode("ascii")
        environment["GIT_CONFIG_COUNT"] = "1"
        environment["GIT_CONFIG_KEY_0"] = "http.extraHeader"
        environment["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {encoded}"
        environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def git_sync(output: Path, message: str, remote: str, branch: str):
    output_path = output.resolve()
    repository = subprocess.run(
        ["git", "-C", str(output_path), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    repository_path = Path(repository).resolve()
    try:
        relative_output = output_path.relative_to(repository_path)
    except ValueError as error:
        raise ValueError(
            "--output-dir must be inside the cloned Git repository when "
            "--sync-github is enabled"
        ) from error
    subprocess.run(
        ["git", "add", "-f", str(relative_output)],
        cwd=repository_path,
        check=True,
    )
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=repository_path
    ).returncode != 0
    if not changed:
        return
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Seq-SCL Kaggle Runner",
            "-c",
            "user.email=1101909@users.noreply.github.com",
            "commit",
            "-m",
            message,
        ],
        cwd=repository_path,
        check=True,
    )
    environment = git_environment()
    last_error = None
    for attempt in range(1, 4):
        result = subprocess.run(
            ["git", "push", remote, f"HEAD:{branch}"],
            cwd=repository_path,
            env=environment,
        )
        if result.returncode == 0:
            print(f"SYNCED GitHub: {message}", flush=True)
            return
        last_error = result.returncode
        print(f"GitHub push attempt {attempt}/3 failed", flush=True)
    raise RuntimeError(f"GitHub push failed with exit code {last_error}")


def summary(rows: list[dict]):
    output = []
    metrics = (
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "NDCG@5",
        "NDCG@10",
        "NDCG@20",
        "MRR@20",
        "Coverage@20",
    )
    for dataset in sorted({row["dataset"] for row in rows}):
        for model in sorted(
            {row["model"] for row in rows if row["dataset"] == dataset}
        ):
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["model"] == model
            ]
            for metric in metrics:
                values = [float(row[metric]) for row in selected]
                output.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "mean": statistics.fmean(values),
                        "std_sample": (
                            statistics.stdev(values) if len(values) > 1 else 0.0
                        ),
                    }
                )
    return output


def run_dataset(
    name: str,
    workspace: Path,
    output: Path,
    config: Config,
    resume: bool,
    sync_github: bool,
    git_remote: str,
    git_branch: str,
    model_keys: tuple[str, ...],
):
    root = workspace / name
    interactions = read_interactions(root, name)
    warm, validation, test, validation_start, test_start = temporal_item_split(
        interactions, config
    )
    train_samples, validation_samples, test_samples, prefix_lengths = build_samples(
        interactions,
        warm,
        validation,
        test,
        validation_start,
        config.max_sequence_length,
    )
    checks = leakage_checks(
        train_samples, validation_samples, test_samples, warm, validation, test
    )
    image, text, original_image_shape, original_text_shape = load_features(
        root, config
    )
    validation_items, test_items = sorted(validation), sorted(test)
    results, losses, run_checks = [], [], {}
    models = []
    for key in model_keys:
        model_name, architecture, lambda_top1 = MODEL_SPECS[key]
        models.append(
            (
                model_name,
                architecture,
                config.lambda_top1 if lambda_top1 == "top1" else lambda_top1,
            )
        )
    for seed in config.seeds:
        run_checks[str(seed)] = {}
        for model_name, architecture, lambda_top1 in models:
            print(f"START {name} {model_name} seed={seed}", flush=True)
            safe = model_name.lower().replace(" ", "_").replace("+", "plus")
            checkpoint_path = (
                output
                / "checkpoints"
                / f"{name}_{safe}_seed_{seed}_latest.pt"
            )

            def on_epoch(row, _checkpoint_path):
                training_row = {"dataset": name, **row}
                upsert_csv(
                    output / "training_log.csv",
                    [training_row],
                    ("dataset", "model", "seed", "epoch"),
                )
                if sync_github:
                    git_sync(
                        output,
                        f"checkpoint: {name} {model_name} seed {seed} "
                        f"epoch {row['epoch']}",
                        git_remote,
                        git_branch,
                    )

            model, epoch_rows, model_check = train_model(
                seed,
                model_name,
                architecture,
                lambda_top1,
                train_samples,
                validation_samples,
                validation_items,
                image,
                text,
                config,
                checkpoint_path,
                resume,
                on_epoch,
            )
            if model.image_features.device.type == "cuda":
                torch.cuda.synchronize()
            test_start = time.perf_counter()
            metrics, predictions = evaluate(
                model, test_samples, test_items, save_predictions=True
            )
            if model.image_features.device.type == "cuda":
                torch.cuda.synchronize()
            test_seconds = time.perf_counter() - test_start
            metrics.update(
                {
                    "dataset": name,
                    "model": model_name,
                    "seed": seed,
                    "best_epoch": model_check["best_epoch"],
                    "train_seconds": model_check["train_seconds"],
                    "test_seconds": test_seconds,
                    "inference_ms_per_user": (
                        1000.0 * test_seconds / len(test_samples)
                    ),
                    "trainable_parameters": model_check[
                        "trainable_parameters"
                    ],
                    "checkpoint_mb": checkpoint_path.stat().st_size / (1024**2),
                    "peak_cuda_memory_mb": model_check[
                        "peak_cuda_memory_mb"
                    ],
                }
            )
            results.append(metrics)
            losses.extend({"dataset": name, **row} for row in epoch_rows)
            run_checks[str(seed)][model_name] = model_check
            write_csv(
                output / f"{name}_{safe}_seed_{seed}_predictions.csv",
                predictions,
            )
            saved_results = upsert_csv(
                output / "results_by_seed.csv",
                [metrics],
                ("dataset", "model", "seed"),
            )
            upsert_csv(
                output / "training_log.csv",
                [{"dataset": name, **row} for row in epoch_rows],
                ("dataset", "model", "seed", "epoch"),
            )
            if len(config.seeds) > 1:
                write_csv(output / "summary.csv", summary(saved_results))
            print(
                f"DONE {name} {model_name} seed={seed} "
                f"R@1={metrics['Recall@1']:.6f} "
                f"NDCG@20={metrics['NDCG@20']:.6f}",
                flush=True,
            )
            if sync_github:
                git_sync(
                    output,
                    f"results: {name} {model_name} seed {seed}",
                    git_remote,
                    git_branch,
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    audit = {
        "dataset": name,
        "config": asdict(config),
        "interactions": len(interactions),
        "original_image_shape": original_image_shape,
        "original_text_shape": original_text_shape,
        "warm_items": len(warm),
        "validation_cold_items": len(validation),
        "test_cold_items": len(test),
        "validation_start_timestamp": validation_start,
        "test_start_timestamp": test_start,
        "train_samples": len(train_samples),
        "validation_users": len(validation_samples),
        "test_users": len(test_samples),
        "prefix_lengths": {
            key: dict(sorted(value.items())) for key, value in prefix_lengths.items()
        },
        "leakage_checks": checks,
        "model_checks": run_checks,
    }
    (output / f"{name}_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    if sync_github:
        git_sync(
            output,
            f"audit: complete {name}",
            git_remote,
            git_branch,
        )
    return results, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", default=["baby", "clothing", "sports"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2022, 2023, 2025])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help=(
            "Models to run. For the multi-interest comparison use: "
            "static multi-item multi-cluster."
        ),
    )
    parser.add_argument("--interest-clusters", type=int, default=3)
    parser.add_argument("--interest-aggregation-tau", type=float, default=0.10)
    parser.add_argument("--cluster-iterations", type=int, default=3)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Directory containing baby/, clothing/, and sports/. "
            "Useful on Kaggle, where data normally lives under /kaggle/input/."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results_mmrec_seq"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each model from its latest per-epoch checkpoint.",
    )
    parser.add_argument(
        "--sync-github",
        action="store_true",
        help="Commit and push output/checkpoints after every epoch.",
    )
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch", default="main")
    args = parser.parse_args()
    config = Config(
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        device=args.device,
        interest_clusters=args.interest_clusters,
        interest_aggregation_tau=args.interest_aggregation_tau,
        cluster_iterations=args.cluster_iterations,
    )
    if config.interest_clusters < 1:
        parser.error("--interest-clusters must be at least 1")
    if config.interest_aggregation_tau <= 0:
        parser.error("--interest-aggregation-tau must be positive")
    if config.cluster_iterations < 1:
        parser.error("--cluster-iterations must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    workspace = args.data_dir.expanduser().resolve()
    all_results, all_losses = [], []
    for name in args.datasets:
        results, losses = run_dataset(
            name,
            workspace,
            args.output_dir,
            config,
            args.resume,
            args.sync_github,
            args.git_remote,
            args.git_branch,
            tuple(args.models),
        )
        all_results.extend(results)
        all_losses.extend(losses)
    print(json.dumps({"summary": summary(all_results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
