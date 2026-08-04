#!/usr/bin/env bash
set -euo pipefail

# Run target-API-excluded pretraining, zero-shot evaluation, and five-fold
# target-domain fine-tuning for the binary and four-class MCC-GCN models.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/revision-three-api}"
FEATURE_ROOT="${FEATURE_ROOT:-$DATA_ROOT/features/covalent}"
OUTDIR="${OUTDIR:-$ROOT/runs/revision-three-api-$(date +%Y%m%d-%H%M%S)}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-}"
TASKS="${TASKS:-binary four-class}"
SEEDS="${SEEDS:-42}"
FOLDS="${FOLDS:-5}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-400}"
FINETUNE_MAX_EPOCHS="${FINETUNE_MAX_EPOCHS:-200}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-16}"
PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
FINETUNE_LR="${FINETUNE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-effective-number}"
EFFECTIVE_NUMBER_BETA="${EFFECTIVE_NUMBER_BETA:-0.9999}"
FINETUNE_LAYERS="${FINETUNE_LAYERS:-0}"
PRETRAIN_PATIENCE="${PRETRAIN_PATIENCE:-30}"
FINETUNE_PATIENCE="${FINETUNE_PATIENCE:-20}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
RESUME="${RESUME:-0}"

required=(
  "$DATA_ROOT/experiment_manifest.json"
  "$DATA_ROOT/four_class_target_physical_pairs.csv"
  "$FEATURE_ROOT/pretrain_train.npz"
  "$FEATURE_ROOT/pretrain_validation.npz"
  "$FEATURE_ROOT/target_all_ab.npz"
  "$FEATURE_ROOT/target_all_ba.npz"
)
for path in "${required[@]}"; do
  test -s "$path" || {
    echo "error: missing experiment input: $path" >&2
    exit 1
  }
done
for fold in $(seq 0 $((FOLDS - 1))); do
  test -s "$FEATURE_ROOT/fold-$fold/train.npz"
  test -s "$FEATURE_ROOT/fold-$fold/test_ab.npz"
  test -s "$FEATURE_ROOT/fold-$fold/test_ba.npz"
done

best_epoch() {
  python - "$1" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))["best_epoch"]
if not isinstance(value, int) or value < 1:
    raise SystemExit(f"invalid best_epoch: {value!r}")
print(value)
PY
}

run_task() {
  local task="$1"
  local model_size="$2"
  local seed="$3"
  local task_dir="$OUTDIR/$task/seed-$seed"
  local pretrain_dir="$task_dir/pretrain"
  local pretrain_tensorboard=()
  mkdir -p "$pretrain_dir"

  if [[ -n "$TENSORBOARD_ROOT" ]]; then
    pretrain_tensorboard=(
      --tensorboard-dir "$TENSORBOARD_ROOT/$task/seed-$seed/pretrain"
    )
  fi

  if [[ "$RESUME" == "1" \
      && -s "$pretrain_dir/best_model.pth" \
      && -s "$pretrain_dir/selection_result.json" ]]; then
    echo "[resume] pretraining checkpoint: $pretrain_dir/best_model.pth"
  else
    python "$ROOT/scripts/train.py" \
      --task "$task" \
      --model-size "$model_size" \
      --data "$FEATURE_ROOT/pretrain_train.npz" \
      --val-data "$FEATURE_ROOT/pretrain_validation.npz" \
      --epochs "$PRETRAIN_EPOCHS" \
      --batch-size "$PRETRAIN_BATCH_SIZE" \
      --lr "$PRETRAIN_LR" \
      --weight-decay "$WEIGHT_DECAY" \
      --seed "$seed" \
      --class-weighting "$CLASS_WEIGHTING" \
      --effective-number-beta "$EFFECTIVE_NUMBER_BETA" \
      --early-stopping-patience "$PRETRAIN_PATIENCE" \
      "${pretrain_tensorboard[@]}" \
      --save-dir "$pretrain_dir"
  fi

  if [[ "$RESUME" == "1" && -s "$pretrain_dir/target_zero_shot_predictions.csv" ]]; then
    echo "[resume] zero-shot predictions"
  else
    python "$ROOT/scripts/evaluate.py" \
      --task "$task" \
      --model-size "$model_size" \
      --model "$pretrain_dir/best_model.pth" \
      --test-data-1 "$FEATURE_ROOT/target_all_ab.npz" \
      --test-data-2 "$FEATURE_ROOT/target_all_ba.npz" \
      --output "$pretrain_dir/target_zero_shot_predictions.csv"
  fi

  python "$ROOT/scripts/summarize_target_api_predictions.py" \
    --task "$task" \
    --target-table "$DATA_ROOT/four_class_target_physical_pairs.csv" \
    --prediction "$pretrain_dir/target_zero_shot_predictions.csv" \
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
    --seed "$seed" \
    --output-dir "$pretrain_dir/target-zero-shot-summary"

  local prediction_args=()
  for fold in $(seq 0 $((FOLDS - 1))); do
    local fold_seed=$((seed + fold))
    local fold_dir="$task_dir/fold-$fold"
    local selection_dir="$fold_dir/selection"
    local final_dir="$fold_dir/final"
    local selection_tensorboard=()
    local final_tensorboard=()
    mkdir -p "$selection_dir" "$final_dir"
    if [[ -n "$TENSORBOARD_ROOT" ]]; then
      selection_tensorboard=(
        --tensorboard-dir \
        "$TENSORBOARD_ROOT/$task/seed-$seed/fold-$fold/selection"
      )
      final_tensorboard=(
        --tensorboard-dir \
        "$TENSORBOARD_ROOT/$task/seed-$seed/fold-$fold/final"
      )
    fi

    if [[ "$RESUME" == "1" && -s "$selection_dir/selection_result.json" ]]; then
      echo "[resume] fold-$fold selection"
    else
      python "$ROOT/scripts/finetune.py" \
        --task "$task" \
        --model-size "$model_size" \
        --mode select \
        --data "$FEATURE_ROOT/fold-$fold/train.npz" \
        --holdout-data "$FEATURE_ROOT/fold-$fold/test_ab.npz" \
        --holdout-data "$FEATURE_ROOT/fold-$fold/test_ba.npz" \
        --pretrained "$pretrain_dir/best_model.pth" \
        --epochs "$FINETUNE_MAX_EPOCHS" \
        --batch-size "$FINETUNE_BATCH_SIZE" \
        --lr "$FINETUNE_LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --seed "$fold_seed" \
        --train-layers "$FINETUNE_LAYERS" \
        --class-weighting "$CLASS_WEIGHTING" \
        --effective-number-beta "$EFFECTIVE_NUMBER_BETA" \
        --early-stopping-patience "$FINETUNE_PATIENCE" \
        "${selection_tensorboard[@]}" \
        --save-dir "$selection_dir"
    fi

    local selected_epoch
    selected_epoch="$(best_epoch "$selection_dir/selection_result.json")"
    if [[ "$RESUME" == "1" && -s "$final_dir/final_FT_model.pth" ]]; then
      echo "[resume] fold-$fold final fit"
    else
      python "$ROOT/scripts/finetune.py" \
        --task "$task" \
        --model-size "$model_size" \
        --mode final-fit \
        --data "$FEATURE_ROOT/fold-$fold/train.npz" \
        --holdout-data "$FEATURE_ROOT/fold-$fold/test_ab.npz" \
        --holdout-data "$FEATURE_ROOT/fold-$fold/test_ba.npz" \
        --pretrained "$pretrain_dir/best_model.pth" \
        --epochs "$selected_epoch" \
        --batch-size "$FINETUNE_BATCH_SIZE" \
        --lr "$FINETUNE_LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --seed "$fold_seed" \
        --train-layers "$FINETUNE_LAYERS" \
        --class-weighting "$CLASS_WEIGHTING" \
        --effective-number-beta "$EFFECTIVE_NUMBER_BETA" \
        "${final_tensorboard[@]}" \
        --save-dir "$final_dir"
    fi

    if [[ "$RESUME" == "1" && -s "$final_dir/target_test_predictions.csv" ]]; then
      echo "[resume] fold-$fold predictions"
    else
      python "$ROOT/scripts/evaluate.py" \
        --task "$task" \
        --model-size "$model_size" \
        --model "$final_dir/final_FT_model.pth" \
        --test-data-1 "$FEATURE_ROOT/fold-$fold/test_ab.npz" \
        --test-data-2 "$FEATURE_ROOT/fold-$fold/test_ba.npz" \
        --seed "$fold_seed" \
        --output "$final_dir/target_test_predictions.csv"
    fi
    prediction_args+=(--prediction "$final_dir/target_test_predictions.csv")
  done

  python "$ROOT/scripts/summarize_target_api_predictions.py" \
    --task "$task" \
    --target-table "$DATA_ROOT/four_class_target_physical_pairs.csv" \
    "${prediction_args[@]}" \
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
    --seed "$seed" \
    --output-dir "$task_dir/oof-summary"
}

mkdir -p "$OUTDIR"
python - "$OUTDIR/experiment_config.json" <<PY
import json
import sys
from pathlib import Path

config = {
    "schema_version": "mcc-gcn-three-api-run-v1",
    "data_root": "$DATA_ROOT",
    "feature_root": "$FEATURE_ROOT",
    "tasks": "$TASKS".split(),
    "seeds": [int(value) for value in "$SEEDS".split()],
    "folds": int("$FOLDS"),
    "pretrain_epochs": int("$PRETRAIN_EPOCHS"),
    "finetune_max_epochs": int("$FINETUNE_MAX_EPOCHS"),
    "class_weighting": "$CLASS_WEIGHTING",
    "effective_number_beta": float("$EFFECTIVE_NUMBER_BETA"),
    "finetune_layers": int("$FINETUNE_LAYERS"),
    "resume": "$RESUME" == "1",
    "tensorboard_root": "$TENSORBOARD_ROOT" or None,
}
Path(sys.argv[1]).write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(config, indent=2, sort_keys=True))
PY

for seed in $SEEDS; do
  for task in $TASKS; do
    case "$task" in
      binary)
        run_task binary small "$seed"
        ;;
      four-class)
        run_task four-class large "$seed"
        ;;
      *)
        echo "error: unsupported task: $task" >&2
        exit 1
        ;;
    esac
  done
done

echo "Three-API experiment complete: $OUTDIR"
