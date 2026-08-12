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
  --epochs 20 --batch-size 256 \
  --run-controlled-content-baselines \
  --run-official-cold-baselines \
  --extended-ablations
```

Official submodules are audit sources and are not required for FIR-MIC itself.
Initialize only an eligible baseline when it is actually scheduled, for example:

```bash
git submodule update --init external/SEMCo
```

For a first GPU check:

```bash
python kaggle/run_gpu.py --datasets baby --seeds 2022 --epochs 1 --batch-size 256
```

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

Additional RQ3 tables include past versus future performance, future-state
quality, history-length groups, and interest-diversity groups. Legacy SEMCo
files without the corrected adapter diagnostics are deliberately excluded from
paper-ready summaries.
