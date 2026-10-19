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

## Quick method check when historical weights are missing

Historical GitHub logs can skip the complete grid search, but JSON logs do not
contain model weights. For a fast Baby seed-2022 check, rebuild only the
GitHub-selected confirmatory model (7 epochs), then run the post-hoc gate:

```python
%cd /kaggle/working/FIR-MIC-MMREC-Cold
!git pull --ff-only origin agent/fix-semco-evaluator
!python -u kaggle/run_gated_quick_test.py \
  --data-dir /kaggle/input/datasets/toanktx/mmrec-cold \
  --dataset baby \
  --seed 2022
```

This reads `optimal_config.json` from the GitHub results branch and skips all
four grid trials, controlled/official baselines, and modality ablations. It
fits only the locked final model, saves a compact learned-only checkpoint, and
then reports original versus gated metrics, including Q4-high.

## Kaggle: run one existing checkpoint

Run these lines in a Kaggle code cell. The leading `!` is required because
these are shell commands, not Python syntax.

```python
%cd /kaggle/working/FIR-MIC-MMREC-Cold
!git pull
!python -u hier_bridge/run_gated_fir_mic_posthoc.py \
  --data-dir /kaggle/input/datasets/toanktx/mmrec-cold \
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

If the checkpoint belongs to an older saved Kaggle notebook version, open that
version's **Output** page and choose **New Notebook** (or add that notebook
output through **Add Input**). The runner automatically searches attached
outputs below `/kaggle/input` and writes new results back to `/kaggle/working`.

You can verify what was attached with:

```python
!find /kaggle/input /kaggle/working -type f -path '*/results/fir_mic/baby/seed_2022/model.pt' -print
```

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
