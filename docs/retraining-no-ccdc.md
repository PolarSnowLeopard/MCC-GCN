# Retraining Without CCDC

This workflow retrains MCC-GCN from precomputed `.npz` graph features. It does
not require `ccdc` or a CCDC license.

## What This Fix Changes

The original NPZ files contain padded `V` and `A` tensors. Older training code
converted the full padded tensors into PyG graphs, so padded zero nodes affected
`global_mean_pool`. The fixed loader uses each sample's `graph_size` and trims
padded rows/columns before PyG conversion.

Because the existing checkpoints were trained with padded zero nodes included,
both stages should be retrained:

1. Pretrain from `HKU_data_5_total_inbalance.npz`.
2. Fine-tune from the new pretrained checkpoint.

## Data Required

Place these files under `data/`:

- `HKU_data_5_total_inbalance.npz`
- `HKU_data_6_FT_minoxidil_balanced_with_exp.npz`
- `HKU_data_6_experiment.npz`
- `HKU_data_6_experiment_1.npz`
- `HKU_data_6_experiment_2.npz`

The `.csv`, `.pkl.gz`, and CCDC files are only needed for feature rebuilding,
not for this workflow.

## Environment

Install the base dependencies only:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Do not install `requirements-ccdc.txt` on machines without a CCDC license.

## Full Retraining

```bash
bash scripts/retrain_from_npz.sh
```

Useful overrides:

```bash
OUTDIR=runs/a100-fixed-loader-001 \
PRETRAIN_DATA=data/HKU_data_5_total_inbalance.npz \
PRETRAIN_EPOCHS=400 \
PRETRAIN_BATCH_SIZE=64 \
FT_EPOCHS=50 \
FT_BATCH_SIZE=16 \
bash scripts/retrain_from_npz.sh
```

Outputs:

- `OUTDIR/pretrain/best_model.pth`
- `OUTDIR/finetune/best_FT_model.pth`
- `OUTDIR/prediction_results.csv`
- `OUTDIR/run.log`

## Smoke Checks

Before a long run:

```bash
python scripts/check_padding_invariance.py \
  --data data/HKU_data_6_experiment_1.npz
```

The expected result is:

```text
PyG graph invariance OK
```

If an old checkpoint is supplied, logit invariance should also pass after the
loader fix:

```bash
python scripts/check_padding_invariance.py \
  --data data/HKU_data_6_experiment_1.npz \
  --model checkpoints/best_FT_model.pth
```
