#!/usr/bin/env bash
set -euo pipefail

# Run corrected binary and four-class experiments without using final holdouts
# for model selection. Provisional runs are isolated and clearly marked.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-final}"
case "$EXPERIMENT_PROFILE" in
  final)
    FEATURE_ROOT="${FEATURE_ROOT:-$ROOT/data/corrected-v2/features/covalent}"
    OUTDIR="${OUTDIR:-$ROOT/runs/corrected-primary-$(date +%Y%m%d-%H%M%S)}"
    finetune_feature="finetune_34.npz"
    finetune_run_name="finetune-34"
    minoxidil_feature="minoxidil_20.npz"
    run_minoxidil_ablation=1
    eligible_for_final_reporting=true
    ;;
  provisional-no-ccdc)
    FEATURE_ROOT="${FEATURE_ROOT:-$ROOT/data/corrected-v2/provisional-no-ccdc/features/covalent}"
    OUTDIR="${OUTDIR:-$ROOT/runs/provisional-no-ccdc-$(date +%Y%m%d-%H%M%S)}"
    finetune_feature="finetune_19_provisional.npz"
    finetune_run_name="finetune-19-provisional"
    minoxidil_feature="minoxidil_5_provisional.npz"
    run_minoxidil_ablation=0
    eligible_for_final_reporting=false
    ;;
  *)
    echo "error: unknown EXPERIMENT_PROFILE=$EXPERIMENT_PROFILE" >&2
    exit 1
    ;;
esac

SEEDS="${SEEDS:-42}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-400}"
FINETUNE_MAX_EPOCHS="${FINETUNE_MAX_EPOCHS:-200}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-16}"
PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
FINETUNE_LR="${FINETUNE_LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-effective-number}"
EFFECTIVE_NUMBER_BETA="${EFFECTIVE_NUMBER_BETA:-0.9999}"
FINETUNE_LAYERS="${FINETUNE_LAYERS:-1}"

required=(
  pretrain_train.npz
  pretrain_validation.npz
  "$finetune_feature"
  "$minoxidil_feature"
  external_64_ab.npz
  external_64_ba.npz
  holdout_50_ab.npz
  holdout_50_ba.npz
)
for name in "${required[@]}"; do
  test -s "$FEATURE_ROOT/$name" || {
    echo "error: missing feature file: $FEATURE_ROOT/$name" >&2
    exit 1
  }
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
  mkdir -p "$task_dir"

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
    --save-dir "$task_dir/pretrain"

  python "$ROOT/scripts/evaluate.py" \
    --task "$task" \
    --model-size "$model_size" \
    --model "$task_dir/pretrain/best_model.pth" \
    --test-data-1 "$FEATURE_ROOT/external_64_ab.npz" \
    --test-data-2 "$FEATURE_ROOT/external_64_ba.npz" \
    --output "$task_dir/pretrain/external_64_predictions.csv"

  python "$ROOT/scripts/finetune.py" \
    --task "$task" \
    --model-size "$model_size" \
    --mode select \
    --data "$FEATURE_ROOT/$finetune_feature" \
    --holdout-data "$FEATURE_ROOT/holdout_50_ab.npz" \
    --pretrained "$task_dir/pretrain/best_model.pth" \
    --epochs "$FINETUNE_MAX_EPOCHS" \
    --batch-size "$FINETUNE_BATCH_SIZE" \
    --lr "$FINETUNE_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --seed "$seed" \
    --train-layers "$FINETUNE_LAYERS" \
    --class-weighting "$CLASS_WEIGHTING" \
    --effective-number-beta "$EFFECTIVE_NUMBER_BETA" \
    --save-dir "$task_dir/$finetune_run_name-selection"

  local ft_epoch
  ft_epoch="$(
    best_epoch "$task_dir/$finetune_run_name-selection/selection_result.json"
  )"
  python "$ROOT/scripts/finetune.py" \
    --task "$task" \
    --model-size "$model_size" \
    --mode final-fit \
    --data "$FEATURE_ROOT/$finetune_feature" \
    --holdout-data "$FEATURE_ROOT/holdout_50_ab.npz" \
    --pretrained "$task_dir/pretrain/best_model.pth" \
    --epochs "$ft_epoch" \
    --batch-size "$FINETUNE_BATCH_SIZE" \
    --lr "$FINETUNE_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --seed "$seed" \
    --train-layers "$FINETUNE_LAYERS" \
    --class-weighting "$CLASS_WEIGHTING" \
    --effective-number-beta "$EFFECTIVE_NUMBER_BETA" \
    --save-dir "$task_dir/$finetune_run_name-final"

  python "$ROOT/scripts/evaluate.py" \
    --task "$task" \
    --model-size "$model_size" \
    --model "$task_dir/$finetune_run_name-final/final_FT_model.pth" \
    --test-data-1 "$FEATURE_ROOT/holdout_50_ab.npz" \
    --test-data-2 "$FEATURE_ROOT/holdout_50_ba.npz" \
    --output "$task_dir/$finetune_run_name-final/holdout_50_predictions.csv"

  if [[ "$run_minoxidil_ablation" != "1" ]]; then
    return
  fi
  python "$ROOT/scripts/finetune.py" \
    --task "$task" \
    --model-size "$model_size" \
    --mode select \
    --data "$FEATURE_ROOT/$minoxidil_feature" \
    --holdout-data "$FEATURE_ROOT/external_64_ab.npz" \
    --pretrained "$task_dir/pretrain/best_model.pth" \
    --epochs "$FINETUNE_MAX_EPOCHS" \
    --batch-size "$FINETUNE_BATCH_SIZE" \
    --lr "$FINETUNE_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --seed "$seed" \
    --train-layers "$FINETUNE_LAYERS" \
    --class-weighting "$CLASS_WEIGHTING" \
    --effective-number-beta "$EFFECTIVE_NUMBER_BETA" \
    --save-dir "$task_dir/minoxidil-20-selection"

  local minoxidil_epoch
  minoxidil_epoch="$(
    best_epoch "$task_dir/minoxidil-20-selection/selection_result.json"
  )"
  python "$ROOT/scripts/finetune.py" \
    --task "$task" \
    --model-size "$model_size" \
    --mode final-fit \
    --data "$FEATURE_ROOT/$minoxidil_feature" \
    --holdout-data "$FEATURE_ROOT/external_64_ab.npz" \
    --pretrained "$task_dir/pretrain/best_model.pth" \
    --epochs "$minoxidil_epoch" \
    --batch-size "$FINETUNE_BATCH_SIZE" \
    --lr "$FINETUNE_LR" \
    --weight-decay "$WEIGHT_DECAY" \
    --seed "$seed" \
    --train-layers "$FINETUNE_LAYERS" \
    --class-weighting "$CLASS_WEIGHTING" \
    --effective-number-beta "$EFFECTIVE_NUMBER_BETA" \
    --save-dir "$task_dir/minoxidil-20-final"

  python "$ROOT/scripts/evaluate.py" \
    --task "$task" \
    --model-size "$model_size" \
    --model "$task_dir/minoxidil-20-final/final_FT_model.pth" \
    --test-data-1 "$FEATURE_ROOT/external_64_ab.npz" \
    --test-data-2 "$FEATURE_ROOT/external_64_ba.npz" \
    --output "$task_dir/minoxidil-20-final/external_64_predictions.csv"
}

mkdir -p "$OUTDIR"
python - "$OUTDIR/experiment_profile.json" <<PY
import json
import sys
from pathlib import Path

profile = {
    "schema_version": "mcc-gcn-experiment-profile-v1",
    "experiment_profile": "$EXPERIMENT_PROFILE",
    "eligible_for_final_reporting": (
        "$eligible_for_final_reporting" == "true"
    ),
    "feature_root": "$FEATURE_ROOT",
    "finetune_feature": "$finetune_feature",
    "minoxidil_ablation_run": bool(int("$run_minoxidil_ablation")),
}
Path(sys.argv[1]).write_text(
    json.dumps(profile, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(profile, indent=2, sort_keys=True))
PY
for seed in $SEEDS; do
  run_task binary small "$seed"
  run_task four-class large "$seed"
done

echo "Corrected experiments complete: $OUTDIR (profile=$EXPERIMENT_PROFILE)"
