#!/usr/bin/env bash
set -euo pipefail

# Compare class-imbalance losses on the frozen pretraining split only.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-provisional-no-ccdc}"
case "$EXPERIMENT_PROFILE" in
  final)
    FEATURE_ROOT="${FEATURE_ROOT:-$ROOT/data/corrected-v2/features/covalent}"
    eligible_for_final_reporting=true
    ;;
  provisional-no-ccdc)
    FEATURE_ROOT="${FEATURE_ROOT:-$ROOT/data/corrected-v2/provisional-no-ccdc/features/covalent}"
    eligible_for_final_reporting=false
    ;;
  *)
    echo "error: unknown EXPERIMENT_PROFILE=$EXPERIMENT_PROFILE" >&2
    exit 1
    ;;
esac

OUTDIR="${OUTDIR:-$ROOT/runs/pretrain-imbalance-$(date +%Y%m%d-%H%M%S)}"
TASKS="${TASKS:-binary four-class}"
CLASS_WEIGHTINGS="${CLASS_WEIGHTINGS:-none inverse-frequency effective-number}"
SEEDS="${SEEDS:-42 43 44}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-400}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EFFECTIVE_NUMBER_BETA="${EFFECTIVE_NUMBER_BETA:-0.9999}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-}"

for name in \
  pretrain_train.npz \
  pretrain_validation.npz \
  external_64_ab.npz \
  external_64_ba.npz; do
  test -s "$FEATURE_ROOT/$name" || {
    echo "error: missing feature file: $FEATURE_ROOT/$name" >&2
    exit 1
  }
done

for task in $TASKS; do
  case "$task" in
    binary|four-class) ;;
    *) echo "error: unknown task: $task" >&2; exit 1 ;;
  esac
done
for weighting in $CLASS_WEIGHTINGS; do
  case "$weighting" in
    none|inverse-frequency|effective-number) ;;
    *) echo "error: unknown class weighting: $weighting" >&2; exit 1 ;;
  esac
done
for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "error: seed must be a non-negative integer: $seed" >&2
    exit 1
  }
done

mkdir -p "$OUTDIR"
"$PYTHON" - \
  "$OUTDIR/experiment_profile.json" \
  "$EXPERIMENT_PROFILE" \
  "$eligible_for_final_reporting" \
  "$FEATURE_ROOT" \
  "$TASKS" \
  "$CLASS_WEIGHTINGS" \
  "$SEEDS" \
  "$TENSORBOARD_ROOT" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    experiment_profile,
    eligible,
    feature_root,
    tasks,
    class_weightings,
    seeds,
    tensorboard_root,
) = sys.argv[1:]
profile = {
    "schema_version": "mcc-gcn-pretrain-imbalance-profile-v1",
    "experiment_profile": experiment_profile,
    "eligible_for_final_reporting": eligible == "true",
    "feature_root": feature_root,
    "tasks": tasks.split(),
    "class_weightings": class_weightings.split(),
    "seeds": [int(seed) for seed in seeds.split()],
    "tensorboard_root": tensorboard_root or None,
}
Path(output).write_text(
    json.dumps(profile, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(profile, indent=2, sort_keys=True))
PY

run_one() {
  local task="$1"
  local model_size="$2"
  local weighting="$3"
  local seed="$4"
  local run_dir="$OUTDIR/$task/$weighting/seed-$seed/pretrain"
  local tensorboard_args=()
  mkdir -p "$run_dir"
  if [[ -n "$TENSORBOARD_ROOT" ]]; then
    tensorboard_args=(
      --tensorboard-dir
      "$TENSORBOARD_ROOT/$task/$weighting/seed-$seed/pretrain"
    )
  fi

  "$PYTHON" "$ROOT/scripts/train.py" \
    --task "$task" \
    --model-size "$model_size" \
    --data "$FEATURE_ROOT/pretrain_train.npz" \
    --val-data "$FEATURE_ROOT/pretrain_validation.npz" \
    --epochs "$PRETRAIN_EPOCHS" \
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
    --batch-size "$PRETRAIN_BATCH_SIZE" \
    --lr "$PRETRAIN_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --seed "$seed" \
    --class-weighting "$weighting" \
    --effective-number-beta "$EFFECTIVE_NUMBER_BETA" \
    ${tensorboard_args[@]+"${tensorboard_args[@]}"} \
    --save-dir "$run_dir"

  "$PYTHON" "$ROOT/scripts/evaluate.py" \
    --task "$task" \
    --model-size "$model_size" \
    --model "$run_dir/best_model.pth" \
    --test-data-1 "$FEATURE_ROOT/external_64_ab.npz" \
    --test-data-2 "$FEATURE_ROOT/external_64_ba.npz" \
    --output "$run_dir/external_64_predictions.csv"
}

for task in $TASKS; do
  model_size=large
  if [[ "$task" == "binary" ]]; then
    model_size=small
  fi
  for weighting in $CLASS_WEIGHTINGS; do
    for seed in $SEEDS; do
      run_one "$task" "$model_size" "$weighting" "$seed"
    done
  done
done

"$PYTHON" "$ROOT/scripts/summarize_pretrain_imbalance.py" \
  --run-dir "$OUTDIR"
echo "Pretraining imbalance ablation complete: $OUTDIR"
