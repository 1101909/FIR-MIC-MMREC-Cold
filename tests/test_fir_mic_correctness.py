import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hier_bridge.run_fir_mic_seed import (
    FIRMIC, batch_rank_loss, make_cache, reset_random_state, router_stay_target,
    row_z, train_model, z,
)
from run_mmrec_seq_scl import Sample

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def small_model():
    reset_random_state(71)
    image = F.normalize(torch.randn(10, 8), dim=-1)
    text = F.normalize(torch.randn(10, 6), dim=-1)
    teacher = torch.softmax(torch.randn(10, 3), -1)
    return FIRMIC(image, text, teacher, 3, d=8, dropout=.2, router_dropout=.1)


def samples():
    return [Sample(i, (i % 5, (i + 1) % 5), 6 + i % 3, (1, 2), 3)
            for i in range(5)]


def test_duplicate_targets_are_positives():
    targets = torch.tensor([4, 4, 9])
    scores = torch.tensor([[.8, .8, .1], [.9, .9, .2], [.1, .1, .8]], requires_grad=True)
    loss = batch_rank_loss(scores, targets)
    expected = -(torch.log_softmax(scores / .1, -1)[:2, :2].logsumexp(-1).sum()
                 + torch.log_softmax(scores / .1, -1)[2, 2]) / 3
    torch.testing.assert_close(loss, expected)
    assert loss < F.cross_entropy(scores / .1, torch.arange(3))
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_unique_targets_match_cross_entropy():
    scores = torch.randn(4, 4)
    torch.testing.assert_close(batch_rank_loss(scores, torch.arange(4)),
                               F.cross_entropy(scores / .1, torch.arange(4)))


def test_standardization_singleton_and_numpy_agree():
    values = torch.tensor([[2.], [3.]], requires_grad=True)
    out = row_z(values)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(values.grad).all()
    matrix = torch.tensor([[1., 3., 6.], [2., 2., 2.]])
    np.testing.assert_allclose(row_z(matrix).numpy(), np.stack([z(row) for row in matrix.numpy()]), atol=1e-6)


def test_router_identical_diffuse_interests_stay():
    q = torch.full((2, 64), 1 / 64.)
    torch.testing.assert_close(router_stay_target(q, q), torch.ones(2))
    torch.testing.assert_close(router_stay_target(torch.tensor([[1., 0.]]),
                                                 torch.tensor([[0., 1.]])), torch.zeros(1))


def test_router_is_invariant_to_other_prefix_lengths():
    model = small_model().eval()
    with torch.no_grad():
        alone = model.routed_components(torch.tensor([[1]]), torch.tensor([1]), torch.tensor([6, 7]))
        mixed = model.routed_components(torch.tensor([[1, 0], [2, 3]]), torch.tensor([1, 2]), torch.tensor([6, 7]))
    torch.testing.assert_close(alone[2][0], mixed[2][0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(alone[5][0], mixed[5][0], atol=1e-6, rtol=1e-5)


def test_validation_does_not_change_confirmatory_training():
    first = small_model()
    second = copy.deepcopy(first)
    def validate(model, epoch):
        # Exercise actual inference and incidental callback RNG consumption.
        make_cache(model, samples(), [6, 7, 8], model.image, model.text, 2, "cpu")
        torch.rand(11)
        return {"NDCG@10": .1, "Recall@10": .1, "MRR@10": .1}
    reset_random_state(18)
    train_model(first, samples(), first.image, first.text, 2, 2, 18, "cpu", epoch_callback=validate)
    reset_random_state(18)
    train_model(second, samples(), second.image, second.text, 2, 2, 18, "cpu")
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_full_feature_baseline_preserves_coordinates(tmp_path):
    from run_mmrec_seq_scl import Config, load_features
    image = np.arange(24, dtype=np.float32).reshape(3, 8) + 1
    text = np.arange(18, dtype=np.float32).reshape(3, 6) + 1
    np.save(tmp_path / "image_feat.npy", image)
    np.save(tmp_path / "text_feat.npy", text)
    actual, _, _, _ = load_features(tmp_path, Config(image_reduced_dim=0))
    np.testing.assert_allclose(actual.numpy(), image / np.linalg.norm(image, axis=1, keepdims=True), atol=1e-6)


def test_identical_results_have_no_statistical_effect():
    from summarize_evidence import METRICS, paired_significance
    rows = [{"dataset": "toy", "seed": seed, "model": model,
             **{metric: .1 for metric in METRICS}}
            for seed in range(3) for model in ("FIR-MIC", "baseline")]
    result = paired_significance(rows, "baseline")
    assert all(row["paired_t_p"] == 1 and row["wilcoxon_p"] == 1 and
               row["effect_size_cohens_dz"] == 0 for row in result)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_significance(rows + [rows[0]], "baseline")


def test_posthoc_gate_keeps_original_when_guardrails_fail():
    from hier_bridge.run_gated_fir_mic_posthoc import select_gate, gated_ranks, original_ranks
    rows = [{"anchor": np.array([1., 0.]), "anchor_z": np.array([1., -1.]),
             "anchor_percentile": np.array([1., 0.]), "auxiliary": np.array([-0.5, 0.5]),
             "target_pos": 0} for _ in range(4)]
    args = SimpleNamespace(gate_w1=[0.], gate_w2=[0.], gate_bias=[-8.],
                           max_overall_ndcg_drop=0., max_easy_ndcg_drop=0.)
    grid, best, _ = select_gate(rows, args)
    assert grid[1]["eligible"] == 0
    assert best["mode"] == "original"
    assert gated_ranks(rows, best["w1"], best["w2"], best["bias"])[0] == original_ranks(rows)
