# Gated-FIR-MIC: inference-only test

This version keeps the trained FIR-MIC checkpoint unchanged and replaces only
the final scoring formula:

```text
alpha(u,c) = sigmoid(w1 * s_anchor(u,c)
                     + w2 * percentile(s_anchor(u,c)) + bias)

S_gated(u,c) = alpha(u,c) * z(s_anchor)
               + (1 - alpha(u,c))
                 * [lambda_z    * z(s_route)
                    + lambda_next * z(s_next)
                    + lambda_fine * z(s_res)]
```

`w1`, `w2`, and `bias` are calibrated by rescoring the validation set only.
There is no backpropagation, optimizer, encoder training, router training, or
test-set model selection. The calibration objective prioritizes the bottom
quartile of validation target-anchor percentile while guarding overall and
easy-subset NDCG@10.

## Kaggle: run one existing checkpoint

Run these lines in a Kaggle code cell. The leading `!` is required because
these are shell commands, not Python syntax.

```python
%cd /kaggle/working/FIR-MIC-MMREC-Cold
!git pull
!python -u hier_bridge/run_gated_fir_mic_posthoc.py \
  --data-dir /kaggle/input/datasets/toanktxd/mmrec-cold \
  --dataset baby \
  --seed 2022
```

The default checkpoint is:

```text
results/fir_mic/baby/seed_2022/model.pt
```

Change `--dataset` and `--seed` to test another preserved checkpoint. The
runner never starts training. It stops immediately with a clear error if
`model.pt` is absent.

## Important checkpoint limitation

The existing GitHub results branch contains small `.json`, `.csv`, `.md`, and
`.txt` artifacts, but intentionally does not contain the large `model.pt`
files. Existing rank CSVs are insufficient to reconstruct a new candidate-level
formula. Therefore this post-hoc run must execute in a Kaggle session/output
where the corresponding `model.pt` still exists.

If Kaggle has reset and removed that checkpoint, the affected seed cannot be
rescored without recovering its original `model.pt` (or retraining that seed).
Do not delete or overwrite a preserved checkpoint.

## Outputs

Files are written to `results/fir_mic/<dataset>/seed_<seed>/gated_posthoc/`:

- `gated_locked_config.json`: validation-locked gate and protocol.
- `gated_validation_grid.csv`: lightweight scoring calibration results.
- `gated_test_metrics.json`: original versus gated aggregate metrics.
- `gated_test_by_gap.csv`: original versus gated Q1--Q4 metrics.
- `gated_test_predictions.csv`: paired per-user ranks and gate diagnostics.

To upload the new small results to the current GitHub results branch after the
run succeeds:

```python
!git add -f results/fir_mic/baby/seed_2022/gated_posthoc
!git commit -m "Add Baby seed 2022 Gated-FIR-MIC post-hoc results"
!git push origin HEAD
```

