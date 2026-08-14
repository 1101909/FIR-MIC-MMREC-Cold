# FIR-MIC on MMREC-COLD

GPU implementation of **Future-Interest Routed Multi-Interest Contrastive
Recommendation** under a strict temporal pure item cold-start protocol.

## Data

Attach the Kaggle dataset `toanktx/mmrec-cold`. The runner discovers both
`/kaggle/input/mmrec-cold` and the legacy
`/kaggle/input/datasets/toanktx/mmrec-cold` mount. Each domain must contain:

```text
baby/baby.inter
baby/image_feat.npy
baby/text_feat.npy
```

The same layout is used for `clothing` and `sports`.

## Kaggle execution

Enable a GPU and Internet in Kaggle, then execute:

```bash
git clone https://github.com/1101909/FIR-MIC-MMREC-Cold.git
cd FIR-MIC-MMREC-Cold
python kaggle/run_gpu.py \
  --datasets baby clothing sports \
  --seeds 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 \
  --epochs 20 \
  --fir-mic-max-epochs 500 \
  --batch-size 256 \
  --grid-intents 32 64 \
  --grid-dims 64 \
  --grid-learning-rates 0.0005 0.001 \
  --grid-weight-decays 0.0001 \
  --grid-dropouts 0.15 \
  --grid-router-dropouts 0.10 \
  --grid-patience 5 \
  --resume \
  --run-controlled-content-baselines \
  --run-official-cold-baselines \
  --extended-ablations
```

For a resumable multi-session run that automatically commits JSON/CSV results
to GitHub after every completed dataset/seed, paste the complete contents of
`kaggle/run_and_push_results.py` into one Kaggle cell. Add a private Kaggle
Secret named `GITHUB_TOKEN` first; it needs Contents read/write access to this
repository. Results are pushed to `kaggle-results-gridsearch-v1`, while large
`.pt` files remain only in the Kaggle working directory. Re-running the same
cell safely skips completed dataset/seed pairs and continues the first
unfinished pair on that same results branch, provided experimental
hyperparameters have not changed.

Official submodules are audit sources and are not required for FIR-MIC itself.
Initialize only an eligible baseline when it is actually scheduled, for example:

```bash
git submodule update --init external/SEMCo
```

For a first GPU check:

```bash
python kaggle/run_gpu.py --datasets baby --seeds 2022 --epochs 1 --batch-size 256
```

## FIR-MIC hyperparameter and epoch selection

For every dataset and seed, FIR-MIC evaluates the Cartesian product supplied by
the `--grid-*` arguments. `--epochs` is the maximum epoch considered in each
trial, and `--grid-patience` stops a trial after consecutive epochs without an
improvement. Selection is lexicographic on validation NDCG@10, Recall@10, then
MRR@10, with the earlier epoch used to break an exact tie.

In the combined Kaggle runner, `--epochs` controls baseline training while
`--fir-mic-max-epochs` independently controls the FIR-MIC search ceiling. The
published runner uses 20 baseline epochs and at most 500 FIR-MIC epochs, with
validation early stopping after 5 non-improving epochs.

Neither cold test interactions nor test metrics are accessed during this
search. After selecting the hyperparameters and epoch, the runner creates a new
model from the same deterministic seed, retrains it for exactly `best_epoch`,
locks the score-level weights on validation, and evaluates the test split once.
Each `results/fir_mic/<dataset>/seed_<seed>/` directory contains:

- `hyperparameter_grid.csv`: every evaluated trial/epoch and validation score;
- `hyperparameter_trials.csv`: the best epoch of every hyperparameter trial;
- `optimal_config.json`: selected architecture, optimizer, epoch, score weights,
  and confirmatory validation metrics;
- `training.csv`: the fresh confirmatory fit;
- `model.pt`: final state dict together with the complete optimal configuration.

With `--resume`, completed runs are skipped and completed search trials are
restored from `search_checkpoints/trial_*.json`. If Kaggle stops during a trial,
only that unfinished trial is repeated; earlier trials are not lost.

The default grid has four trials (2 intent counts × 2 learning rates). Expand
the list arguments only when the available GPU budget permits it.

## Fair-baseline policy

Official baseline repositories are pinned as Git submodules. Their architecture
is never replaced. A format-only adapter is allowed; a new cold-item decoder is
not. SASRec and ReaRec are excluded because their official ID models cannot
score unseen cold IDs. TGH is excluded from the main table because its official
split and T5-XL representation protocol differ. CLCRec cannot accept this
dataset without editing hard-coded dataset cardinalities. GoRec's GitHub data
does not include the interactions needed to reconstruct this split. SEMCo is
methodologically eligible, but its official script has no Baby preset; it must
be tuned on validation before entering a final table.

The `static`, `multi-item`, and `multi-cluster` runs are explicitly labelled
controlled content baselines, not official reproductions of another GitHub
repository.

## RQ3 semantic-gap analysis

After the per-seed runs, `kaggle/run_gpu.py` automatically evaluates whether
FIR-MIC's gain over Multi-interest per-item grows with history-to-target
semantic gap. The fixed content-only gap from the earlier pilot is:

```text
1 - max_history(0.2 * cosine(image) + 0.8 * cosine(text))
```

Samples are divided into deterministic balanced quartiles (`Q1-low` through
`Q4-high`). Outputs are saved under `results/rq3/<dataset>/`:

- `semantic_gap_assignments.csv`: model-independent sample groups;
- `metrics_by_seed.csv` and `summary.csv`: group metrics and mean plus SD;
- `paired_fir_mic_vs_per_item.csv`: paired per-seed improvements;
- `rq3_report.json`: exact sign-flip tests and the primary
  `gain(Q4) - gain(Q1)` interaction contrast.

The gap uses intrinsic cold-item content only for post-hoc evaluation. Gap
assignments and cold interactions never enter training or model selection.

## Paper evidence tables

Use `--extended-ablations` for the confirmatory run. It retrains image-only and
text-only FIR-MIC variants rather than masking modalities after training. Each
seed additionally writes component ranks, past/future target-intent cosine,
protocol statistics, transition statistics, and efficiency measurements.

The pipeline finishes by running `scripts/summarize_evidence.py`. Paper-ready
cross-seed tables are saved under `results/evidence/`:

- `main_raw_by_seed.csv`, `main_mean_std.csv`, and `main_gains.csv`;
- `ablation_mean_std.csv`;
- `statistical_significance.csv` with paired t-test, Wilcoxon, Holm correction,
  confidence intervals, and effect sizes;
- `protocol_mean_std.csv`, `transition_mean_std.csv`, and
  `efficiency_mean_std.csv`.
- `optimal_configs_by_seed.csv` and `hyperparameter_trials_by_seed.csv` for a
  complete audit of model and epoch selection.

Additional RQ3 tables include past versus future performance, future-state
quality, history-length groups, and interest-diversity groups. Legacy SEMCo
files without the corrected adapter diagnostics are deliberately excluded from
paper-ready summaries.
