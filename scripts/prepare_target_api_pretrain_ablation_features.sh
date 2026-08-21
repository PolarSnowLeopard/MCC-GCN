#!/usr/bin/env bash
set -euo pipefail

# Freeze task-specific undersampled source tables and build RDKit features.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/revision-three-api}"
ABLATION_ROOT="${ABLATION_ROOT:-$DATA_ROOT/pretrain-ablation}"
SUBSET_SEED="${SUBSET_SEED:-42}"
FORCE="${FORCE:-0}"

TRAIN_PHYSICAL="$DATA_ROOT/pretrain-split/train_physical_pairs.csv"
TABLE_ROOT="$ABLATION_ROOT/tables"
FEATURE_ROOT="$ABLATION_ROOT/features/covalent"

test -s "$TRAIN_PHYSICAL" || {
  echo "error: missing physical-pair training table: $TRAIN_PHYSICAL" >&2
  exit 1
}

mkdir -p "$TABLE_ROOT" "$FEATURE_ROOT"
"$PYTHON" "$ROOT/scripts/build_target_api_pretrain_ablation.py" \
  --train-physical "$TRAIN_PHYSICAL" \
  --output-dir "$TABLE_ROOT" \
  --seed "$SUBSET_SEED"

build_features() {
  local input="$1"
  local output="$2"
  if [[ "$FORCE" != "1" && -s "$output" && -s "${output%.npz}.manifest.json" ]]; then
    echo "[skip] feature file already exists: $output"
    return
  fi
  "$PYTHON" "$ROOT/scripts/build_features.py" \
    --input "$input" \
    --output "$output" \
    --feature-source rdkit_smiles \
    --rdkit-coordinate-mode 2d \
    --adjacency-type OnlyCovalentBond \
    --storage-format packed-sparse
}

build_features \
  "$TABLE_ROOT/binary_undersampled_ordered.csv" \
  "$FEATURE_ROOT/binary_undersampled_train.npz"
build_features \
  "$TABLE_ROOT/four_class_undersampled_ordered.csv" \
  "$FEATURE_ROOT/four_class_undersampled_train.npz"

echo "Target-API pretraining ablation features ready: $FEATURE_ROOT"
