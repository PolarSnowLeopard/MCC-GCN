#!/usr/bin/env bash
set -euo pipefail

# End-to-end retraining from precomputed NPZ features. This path does not
# require CCDC because it never rebuilds graph features from raw structures.

OUTDIR="${OUTDIR:-runs/retrain-fixed-loader-$(date +%Y%m%d-%H%M%S)}"
LOG="$OUTDIR/run.log"

PRETRAIN_DATA="${PRETRAIN_DATA:-data/HKU_data_5_total_inbalance.npz}"
FT_DATA="${FT_DATA:-data/HKU_data_6_FT_minoxidil_balanced_with_exp.npz}"
VAL_DATA="${VAL_DATA:-data/HKU_data_6_experiment.npz}"
TEST_DATA_1="${TEST_DATA_1:-data/HKU_data_6_experiment_1.npz}"
TEST_DATA_2="${TEST_DATA_2:-data/HKU_data_6_experiment_2.npz}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-400}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
FT_EPOCHS="${FT_EPOCHS:-50}"
FT_BATCH_SIZE="${FT_BATCH_SIZE:-16}"

PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
FT_LR="${FT_LR:-3e-4}"
SEED="${SEED:-42}"
FT_SEED="${FT_SEED:-12}"

mkdir -p "$OUTDIR"

run() {
  echo "+ $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

{
  echo "outdir=$OUTDIR"
  echo "pretrain_data=$PRETRAIN_DATA"
  echo "ft_data=$FT_DATA"
  echo "val_data=$VAL_DATA"
  echo "test_data_1=$TEST_DATA_1"
  echo "test_data_2=$TEST_DATA_2"
  echo "pretrain_epochs=$PRETRAIN_EPOCHS"
  echo "ft_epochs=$FT_EPOCHS"
  echo "seed=$SEED"
  echo "ft_seed=$FT_SEED"
} | tee "$LOG"

run python scripts/check_padding_invariance.py \
  --data "$TEST_DATA_1" \
  --samples 10

run python scripts/train.py \
  --data "$PRETRAIN_DATA" \
  --epochs "$PRETRAIN_EPOCHS" \
  --batch-size "$PRETRAIN_BATCH_SIZE" \
  --lr "$PRETRAIN_LR" \
  --seed "$SEED" \
  --save-dir "$OUTDIR/pretrain"

run python scripts/finetune.py \
  --data "$FT_DATA" \
  --val-data "$VAL_DATA" \
  --pretrained "$OUTDIR/pretrain/best_model.pth" \
  --epochs "$FT_EPOCHS" \
  --batch-size "$FT_BATCH_SIZE" \
  --lr "$FT_LR" \
  --seed "$FT_SEED" \
  --save-dir "$OUTDIR/finetune"

run python scripts/evaluate.py \
  --model "$OUTDIR/finetune/best_FT_model.pth" \
  --test-data-1 "$TEST_DATA_1" \
  --test-data-2 "$TEST_DATA_2" \
  --output "$OUTDIR/prediction_results.csv"

echo "Retraining complete: $OUTDIR" | tee -a "$LOG"
