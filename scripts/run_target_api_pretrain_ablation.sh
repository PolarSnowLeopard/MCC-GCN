#!/usr/bin/env bash
set -euo pipefail

# Compare all-data class weighting with task-specific random undersampling.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/revision-three-api}"
FEATURE_ROOT="${FEATURE_ROOT:-$DATA_ROOT/features/covalent}"
ABLATION_FEATURE_ROOT="${ABLATION_FEATURE_ROOT:-$DATA_ROOT/pretrain-ablation/features/covalent}"
ABLATION_TABLE_ROOT="${ABLATION_TABLE_ROOT:-$DATA_ROOT/pretrain-ablation/tables}"
OUTDIR="${OUTDIR:-$ROOT/runs/revision-three-api-pretrain-ablation-$(date +%Y%m%d-%H%M%S)}"
TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-}"
TASKS="${TASKS:-binary four-class}"
STRATEGIES="${STRATEGIES:-full-effective-number undersampled-uniform}"
SEEDS="${SEEDS:-42 43 44}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-400}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-64}"
PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EFFECTIVE_NUMBER_BETA="${EFFECTIVE_NUMBER_BETA:-0.9999}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-30}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-2000}"
RESUME="${RESUME:-0}"

required=(
  "$DATA_ROOT/four_class_target_physical_pairs.csv"
  "$FEATURE_ROOT/pretrain_train.npz"
  "$FEATURE_ROOT/pretrain_train.manifest.json"
  "$FEATURE_ROOT/pretrain_validation.npz"
  "$FEATURE_ROOT/target_all_ab.npz"
  "$FEATURE_ROOT/target_all_ba.npz"
  "$ABLATION_TABLE_ROOT/manifest.json"
  "$ABLATION_FEATURE_ROOT/binary_undersampled_train.npz"
  "$ABLATION_FEATURE_ROOT/binary_undersampled_train.manifest.json"
  "$ABLATION_FEATURE_ROOT/four_class_undersampled_train.npz"
  "$ABLATION_FEATURE_ROOT/four_class_undersampled_train.manifest.json"
)
for path in "${required[@]}"; do
  test -s "$path" || {
    echo "error: missing ablation input: $path" >&2
    exit 1
  }
done

for task in $TASKS; do
  case "$task" in
    binary|four-class) ;;
    *) echo "error: unsupported task: $task" >&2; exit 1 ;;
  esac
done
for strategy in $STRATEGIES; do
  case "$strategy" in
    full-effective-number|undersampled-uniform) ;;
    *) echo "error: unsupported strategy: $strategy" >&2; exit 1 ;;
  esac
done
for seed in $SEEDS; do
  [[ "$seed" =~ ^[0-9]+$ ]] || {
    echo "error: seed must be a non-negative integer: $seed" >&2
    exit 1
  }
done

training_input() {
  local task="$1"
  local strategy="$2"
  if [[ "$strategy" == "full-effective-number" ]]; then
    printf '%s\n' "$FEATURE_ROOT/pretrain_train.npz"
  elif [[ "$task" == "binary" ]]; then
    printf '%s\n' "$ABLATION_FEATURE_ROOT/binary_undersampled_train.npz"
  else
    printf '%s\n' "$ABLATION_FEATURE_ROOT/four_class_undersampled_train.npz"
  fi
}

class_weighting() {
  if [[ "$1" == "full-effective-number" ]]; then
    printf '%s\n' effective-number
  else
    printf '%s\n' none
  fi
}

mkdir -p "$OUTDIR"
"$PYTHON" - \
  "$OUTDIR/experiment_profile.json" \
  "$DATA_ROOT" \
  "$FEATURE_ROOT" \
  "$ABLATION_FEATURE_ROOT" \
  "$ABLATION_TABLE_ROOT/manifest.json" \
  "$TASKS" \
  "$STRATEGIES" \
  "$SEEDS" \
  "$PRETRAIN_EPOCHS" \
  "$PRETRAIN_BATCH_SIZE" \
  "$PRETRAIN_LR" \
  "$WEIGHT_DECAY" \
  "$EFFECTIVE_NUMBER_BETA" \
  "$EARLY_STOPPING_PATIENCE" \
  "$BOOTSTRAP_REPLICATES" \
  "$TENSORBOARD_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    output,
    data_root,
    feature_root,
    ablation_feature_root,
    subset_manifest_path,
    tasks,
    strategies,
    seeds,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    effective_number_beta,
    patience,
    bootstrap_replicates,
    tensorboard_root,
) = sys.argv[1:]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_description(path):
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "physical_pairs": int(manifest["physical_pairs"]),
        "ordered_rows": int(manifest["ordered_rows"]),
        "label_counts": manifest["label_counts"],
    }


full = feature_description(Path(feature_root) / "pretrain_train.npz")
binary = feature_description(
    Path(ablation_feature_root) / "binary_undersampled_train.npz"
)
four_class = feature_description(
    Path(ablation_feature_root) / "four_class_undersampled_train.npz"
)
subset_manifest = json.loads(Path(subset_manifest_path).read_text(encoding="utf-8"))
profile = {
    "schema_version": "mcc-gcn-target-pretrain-undersampling-run-v1",
    "data_root": data_root,
    "tasks": tasks.split(),
    "strategies": {
        "full-effective-number": {
            "class_weighting": "effective-number",
            "features_by_task": {"binary": full, "four-class": full},
        },
        "undersampled-uniform": {
            "class_weighting": "none",
            "features_by_task": {
                "binary": binary,
                "four-class": four_class,
            },
        },
    },
    "enabled_strategies": strategies.split(),
    "seeds": [int(value) for value in seeds.split()],
    "subset_manifest": {
        "path": subset_manifest_path,
        "sha256": sha256(subset_manifest_path),
        "subset_seed": int(subset_manifest["subset_seed"]),
    },
    "validation_features": feature_description(
        Path(feature_root) / "pretrain_validation.npz"
    ),
    "target_features": {
        "forward": feature_description(Path(feature_root) / "target_all_ab.npz"),
        "reverse": feature_description(Path(feature_root) / "target_all_ba.npz"),
    },
    "training": {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "effective_number_beta": float(effective_number_beta),
        "early_stopping_patience": int(patience),
    },
    "bootstrap_replicates": int(bootstrap_replicates),
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
  local strategy="$3"
  local seed="$4"
  local data
  local weighting
  local run_dir="$OUTDIR/$task/$strategy/seed-$seed/pretrain"
  local tensorboard_args=()
  data="$(training_input "$task" "$strategy")"
  weighting="$(class_weighting "$strategy")"
  mkdir -p "$run_dir"

  if [[ -n "$TENSORBOARD_ROOT" ]]; then
    tensorboard_args=(
      --tensorboard-dir "$TENSORBOARD_ROOT/$task/$strategy/seed-$seed/pretrain"
    )
  fi

  if [[ "$RESUME" == "1" \
      && -s "$run_dir/best_model.pth" \
      && -s "$run_dir/selection_result.json" ]]; then
    echo "[resume] checkpoint: $run_dir/best_model.pth"
  else
    "$PYTHON" "$ROOT/scripts/train.py" \
      --task "$task" \
      --model-size "$model_size" \
      --data "$data" \
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
  fi

  if [[ "$RESUME" == "1" && -s "$run_dir/target_predictions.csv" ]]; then
    echo "[resume] predictions: $run_dir/target_predictions.csv"
  else
    "$PYTHON" "$ROOT/scripts/evaluate.py" \
      --task "$task" \
      --model-size "$model_size" \
      --model "$run_dir/best_model.pth" \
      --test-data-1 "$FEATURE_ROOT/target_all_ab.npz" \
      --test-data-2 "$FEATURE_ROOT/target_all_ba.npz" \
      --output "$run_dir/target_predictions.csv"
  fi

  "$PYTHON" "$ROOT/scripts/summarize_target_api_predictions.py" \
    --task "$task" \
    --target-table "$DATA_ROOT/four_class_target_physical_pairs.csv" \
    --prediction "$run_dir/target_predictions.csv" \
    --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
    --seed "$seed" \
    --output-dir "$run_dir/target-summary"
}

for task in $TASKS; do
  model_size=large
  if [[ "$task" == "binary" ]]; then
    model_size=small
  fi
  for strategy in $STRATEGIES; do
    for seed in $SEEDS; do
      run_one "$task" "$model_size" "$strategy" "$seed"
    done
  done
done

"$PYTHON" "$ROOT/scripts/summarize_target_api_pretrain_ablation.py" \
  --run-dir "$OUTDIR"
echo "Target-API pretraining ablation complete: $OUTDIR"
