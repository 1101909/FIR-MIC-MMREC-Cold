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
git clone --recurse-submodules https://github.com/1101909/FIR-MIC-MMREC-Cold.git
cd FIR-MIC-MMREC-Cold
python kaggle/run_gpu.py \
  --datasets baby clothing sports \
  --seeds 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 \
  --epochs 20 --batch-size 256 \
  --run-controlled-content-baselines
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
