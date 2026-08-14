import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from hier_bridge.run_fir_mic_seed import grid_search
from run_mmrec_seq_scl import Sample

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_evidence import collect_hyperparameters, read_csv, write_csv


def test_write_csv_accepts_sparse_model_specific_fields(tmp_path: Path):
    destination = tmp_path / "efficiency.csv"
    write_csv(destination, [
        {"model": "Static", "train_seconds_mean": 1.0},
        {
            "model": "FIR-MIC",
            "train_seconds_mean": 2.0,
            "hyperparameter_search_seconds_mean": 3.0,
        },
    ])

    rows = read_csv(destination)
    assert rows[0]["hyperparameter_search_seconds_mean"] == ""
    assert rows[1]["hyperparameter_search_seconds_mean"] == "3.0"


def normalized(rng, rows, columns):
    values = rng.normal(size=(rows, columns)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_grid_search_selects_and_checkpoints_without_test_data(tmp_path: Path):
    rng = np.random.default_rng(7)
    image_np = normalized(rng, 12, 8)
    text_np = normalized(rng, 12, 6)
    image = torch.from_numpy(image_np)
    text = torch.from_numpy(text_np)
    train = [
        Sample(user, (user % 6, (user + 1) % 6), (user + 2) % 8, (1, 2), 3)
        for user in range(8)
    ]
    validation = [
        Sample(20, (0, 1), 8, (1, 2), 3),
        Sample(21, (2, 3), 9, (1, 2), 3),
    ]
    args = SimpleNamespace(
        seed=2022, grid_intents=[2], grid_dims=[8],
        grid_learning_rates=[1e-3], grid_weight_decays=[1e-4],
        grid_dropouts=[0.0], grid_router_dropouts=[0.0],
        grid_patience=1, epochs=1, batch_size=4,
        output_dir=tmp_path, resume=False,
    )

    optimal, epoch_rows, trial_rows = grid_search(
        args, image_np, text_np, image, text, set(range(8)), train,
        validation, {8, 9}, torch.device("cpu"),
    )

    assert optimal["best_epoch"] == 1
    assert optimal["selection_split"] == "validation only"
    assert len(epoch_rows) == len(trial_rows) == 1
    assert not any("test" in key.lower() for key in epoch_rows[0])
    assert (tmp_path / "search_checkpoints" / "trial_0001.json").exists()

    args.resume = True
    resumed, resumed_epochs, resumed_trials = grid_search(
        args, image_np, text_np, image, text, set(range(8)), train,
        validation, {8, 9}, torch.device("cpu"),
    )
    assert resumed["best_epoch"] == optimal["best_epoch"]
    assert resumed_epochs == epoch_rows
    assert resumed_trials == trial_rows

    run_dir = tmp_path / "results" / "fir_mic" / "baby" / "seed_2022"
    run_dir.mkdir(parents=True)
    (run_dir / "optimal_config.json").write_text(
        json.dumps(optimal), encoding="utf8"
    )
    source_trials = tmp_path / "hyperparameter_trials.csv"
    (run_dir / "hyperparameter_trials.csv").write_text(
        source_trials.read_text(encoding="utf8"), encoding="utf8"
    )
    selected, collected_trials = collect_hyperparameters(tmp_path / "results")
    assert selected[0]["best_epoch"] == 1
    assert selected[0]["selection_split"] == "validation only"
    assert collected_trials[0]["dataset"] == "baby"
