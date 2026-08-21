#!/usr/bin/env bash
set -euo pipefail

# Fine-tune fixed pretrained checkpoints on nested target-domain subsets.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/revision-three-api}"
FEATURE_ROOT="${FEATURE_ROOT:-$DATA_ROOT/features/covalent}"
OUTDIR="${OUTDIR:-$ROOT/runs/target-api-learning-curve-$(date +%Y%m%d-%H%M%S)}"
SUBSET_ROOT="${SUBSET_ROOT:-$OUTDIR/subsets}"
SOURCE_RUNS="${SOURCE_RUNS:-}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-}"
TASKS="${TASKS:-binary four-class}"
SEEDS="${SEEDS:-42 43 44}"
SIZES="${SIZES:-8 16 32 48}"
FOLDS="${FOLDS:-5}"
SUBSET_SEED="${SUBSET_SEED:-42}"
FINETUNE_MAX_EPOCHS="${FINETUNE_MAX_EPOCHS:-200}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-16}"
FINETUNE_LR="${FINETUNE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-effective-number}"
EFFECTIVE_NUMBER_BETA="${EFFECTIVE_NUMBER_BETA:-0.9999}"
FINETUNE_LAYERS="${FINETUNE_LAYERS:-0}"
FINETUNE_PATIENCE="${FINETUNE_PATIENCE:-20}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
RESUME="${RESUME:-0}"

if [[ -z "$SOURCE_RUNS" ]]; then
  echo "error: SOURCE_RUNS must list the completed three-API run roots" >&2
  exit 1
fi

required=(
  "$DATA_ROOT/experiment_manifest.json"
  "$DATA_ROOT/four_class_target_physical_pairs.csv"
  "$FEATURE_ROOT/fold-0/train.npz"
)
for path in "${required[@]}"; do
  test -s "$path" || {
    echo "error: missing learning-curve input: $path" >&2
    exit 1
  }
done

sizes_csv="$(printf '%s' "$SIZES" | tr ' ' ',')"
python "$ROOT/scripts/build_target_api_learning_curve.py" \
  --data-root "$DATA_ROOT" \
  --output-root "$SUBSET_ROOT" \
  --sizes "$sizes_csv" \
  --folds "$FOLDS" \
  --seed "$SUBSET_SEED"

find_checkpoint() {
  local task="$1"
  local seed="$2"
  local found=""
  local source path
  for source in $SOURCE_RUNS; do
    path="$source/$task/seed-$seed/pretrain/best_model.pth"
    if [[ -s "$path" ]]; then
      if [[ -n "$found" ]]; then
        echo "error: duplicate checkpoint for $task seed $seed" >&2
        return 1
      fi
      found="$path"
    fi
  done
  if [[ -z "$found" ]]; then
    echo "error: checkpoint not found for $task seed $seed" >&2
    return 1
  fi
  printf '%s\n' "$found"
}

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

validation_fraction() {
  case "$1" in
    8) printf '0.5\n' ;;
    16) printf '0.25\n' ;;
    *) printf '0.2\n' ;;
  esac
}

run_task_size() {
  local task="$1"
  local model_size="$2"
  local seed="$3"
  local size="$4"
  local pretrained="$5"
  local size_dir="$OUTDIR/$task/seed-$seed/size-$size"
  local prediction_args=()
  local fold

  for fold in $(seq 0 $((FOLDS - 1))); do
    local fold_seed=$((seed + fold))
    local subset="$SUBSET_ROOT/fold-$fold/size-$size.csv"
    local feature_dir="$FEATURE_ROOT/fold-$fold"
    local fold_dir="$size_dir/fold-$fold"
    local selection_dir="$fold_dir/selection"
    local final_dir="$fold_dir/final"
    local selected_epoch
    local val_fraction
    local selection_tensorboard=()
    local final_tensorboard=()
    val_fraction="$(validation_fraction "$size")"
    mkdir -p "$selection_dir" "$final_dir"

    if [[ -n "$TENSORBOARD_ROOT" ]]; then
      selection_tensorboard=(
        --tensorboard-dir
        "$TENSORBOARD_ROOT/$task/seed-$seed/size-$size/fold-$fold/selection"
      )
      final_tensorboard=(
        --tensorboard-dir
        "$TENSORBOARD_ROOT/$task/seed-$seed/size-$size/fold-$fold/final"
      )
    fi

    if [[ "$RESUME" == "1" && -s "$selection_dir/selection_result.json" ]]; then
      echo "[resume] $task seed-$seed size-$size fold-$fold selection"
    else
      python "$ROOT/scripts/finetune.py" \
        --task "$task" \
        --model-size "$model_size" \
        --mode select \
        --data "$feature_dir/train.npz" \
        --pair-subset-manifest "$subset" \
        --holdout-data "$feature_dir/test_ab.npz" \
        --holdout-data "$feature_dir/test_ba.npz" \
        --pretrained "$pretrained" \
        --epochs "$FINETUNE_MAX_EPOCHS" \
        --batch-size "$FINETUNE_BATCH_SIZE" \
        --lr "$FINETUNE_LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --seed "$fold_seed" \
        --train-layers "$FINETUNE_LAYERS" \
        --validation-fraction "$val_fraction" \
        --class-weighting "$CLASS_WEIGHTING" \
        --effective-number-beta "$EFFECTIVE_NUMBER_BETA" \
        --early-stopping-patience "$FINETUNE_PATIENCE" \
        "${selection_tensorboard[@]}" \
        --save-dir "$selection_dir"
    fi

    selected_epoch="$(best_epoch "$selection_dir/selection_result.json")"
    if [[ "$RESUME" == "1" && -s "$final_dir/final_FT_model.pth" ]]; then
      echo "[resume] $task seed-$seed size-$size fold-$fold final"
    else
      python "$ROOT/scripts/finetune.py" \
        --task "$task" \
        --model-size "$model_size" \
        --mode final-fit \
        --data "$feature_dir/train.npz" \
        --pair-subset-manifest "$subset" \
        --holdout-data "$feature_dir/test_ab.npz" \
        --holdout-data "$feature_dir/test_ba.npz" \
        --pretrained "$pretrained" \
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
      echo "[resume] $task seed-$seed size-$size fold-$fold predictions"
    else
      python "$ROOT/scripts/evaluate.py" \
        --task "$task" \
        --model-size "$model_size" \
        --model "$final_dir/final_FT_model.pth" \
        --test-data-1 "$feature_dir/test_ab.npz" \
        --test-data-2 "$feature_dir/test_ba.npz" \
        --seed "$fold_seed" \
        --output "$final_dir/target_test_predictions.csv"
    fi
    prediction_args+=(
      --prediction "$final_dir/target_test_predictions.csv"
    )
  done

  python "$ROOT/scripts/summarize_target_api_predictions.py" \
    --task "$task" \
    --target-table "$DATA_ROOT/four_class_target_physical_pairs.csv" \
    "${prediction_args[@]}" \
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
    --seed "$seed" \
    --output-dir "$size_dir/oof-summary"
}

mkdir -p "$OUTDIR"
python - "$OUTDIR/experiment_config.json" <<PY
import json
import sys
from pathlib import Path

config = {
    "schema_version": "mcc-gcn-target-api-learning-curve-run-v1",
    "data_root": "$DATA_ROOT",
    "feature_root": "$FEATURE_ROOT",
    "source_runs": "$SOURCE_RUNS".split(),
    "tasks": "$TASKS".split(),
    "seeds": [int(value) for value in "$SEEDS".split()],
    "sizes": [int(value) for value in "$SIZES".split()],
    "folds": int("$FOLDS"),
    "subset_seed": int("$SUBSET_SEED"),
    "subset_strategy": "nested_class_balanced_physical_pairs",
    "finetune_max_epochs": int("$FINETUNE_MAX_EPOCHS"),
    "finetune_layers": int("$FINETUNE_LAYERS"),
    "class_weighting": "$CLASS_WEIGHTING",
    "resume": "$RESUME" == "1",
}
Path(sys.argv[1]).write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(config, indent=2, sort_keys=True))
PY

for seed in $SEEDS; do
  for task in $TASKS; do
    checkpoint="$(find_checkpoint "$task" "$seed")"
    for size in $SIZES; do
      case "$task" in
        binary) run_task_size binary small "$seed" "$size" "$checkpoint" ;;
        four-class) run_task_size four-class large "$seed" "$size" "$checkpoint" ;;
        *) echo "error: unsupported task: $task" >&2; exit 1 ;;
      esac
    done
  done
done

python "$ROOT/scripts/summarize_target_api_runs.py" \
  --run-dir "$OUTDIR" \
  --output-dir "$OUTDIR/multiseed-summary"

echo "Target-API learning curve complete: $OUTDIR"
