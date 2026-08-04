#!/usr/bin/env bash
set -euo pipefail

# Build CCDC-free RDKit features for the frozen three-API experiment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/revision-three-api}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/features/covalent}"
FOLDS="${FOLDS:-5}"
FORCE="${FORCE:-0}"

required=(
  "$DATA_ROOT/experiment_manifest.json"
  "$DATA_ROOT/four_class_target_physical_pairs.csv"
  "$DATA_ROOT/pretrain-split/train_ordered_pairs.csv"
  "$DATA_ROOT/pretrain-split/validation_ordered_pairs.csv"
  "$DATA_ROOT/folds/fold_assignments.csv"
)
for path in "${required[@]}"; do
  test -s "$path" || {
    echo "error: missing or empty input: $path" >&2
    exit 1
  }
done

mkdir -p "$OUTPUT_ROOT/tables"

python "$ROOT/scripts/augment_pair_orders.py" \
  --input "$DATA_ROOT/four_class_target_physical_pairs.csv" \
  --forward-output "$OUTPUT_ROOT/tables/target_all_ab.csv" \
  --reverse-output "$OUTPUT_ROOT/tables/target_all_ba.csv"

for fold in $(seq 0 $((FOLDS - 1))); do
  fold_input="$DATA_ROOT/folds/fold-$fold"
  fold_tables="$OUTPUT_ROOT/tables/fold-$fold"
  test -s "$fold_input/train_physical_pairs.csv"
  test -s "$fold_input/test_physical_pairs.csv"
  mkdir -p "$fold_tables"
  python "$ROOT/scripts/augment_pair_orders.py" \
    --input "$fold_input/train_physical_pairs.csv" \
    --output "$fold_tables/train_ordered.csv"
  python "$ROOT/scripts/augment_pair_orders.py" \
    --input "$fold_input/test_physical_pairs.csv" \
    --forward-output "$fold_tables/test_ab.csv" \
    --reverse-output "$fold_tables/test_ba.csv"
done

build_features() {
  local input="$1"
  local output="$2"
  if [[ "$FORCE" != "1" && -s "$output" && -s "${output%.npz}.manifest.json" ]]; then
    echo "[skip] feature file already exists: $output"
    return
  fi
  python "$ROOT/scripts/build_features.py" \
    --input "$input" \
    --output "$output" \
    --feature-source rdkit_smiles \
    --rdkit-coordinate-mode 2d \
    --adjacency-type OnlyCovalentBond
}

build_features \
  "$DATA_ROOT/pretrain-split/train_ordered_pairs.csv" \
  "$OUTPUT_ROOT/pretrain_train.npz"
build_features \
  "$DATA_ROOT/pretrain-split/validation_ordered_pairs.csv" \
  "$OUTPUT_ROOT/pretrain_validation.npz"
build_features \
  "$OUTPUT_ROOT/tables/target_all_ab.csv" \
  "$OUTPUT_ROOT/target_all_ab.npz"
build_features \
  "$OUTPUT_ROOT/tables/target_all_ba.csv" \
  "$OUTPUT_ROOT/target_all_ba.npz"

for fold in $(seq 0 $((FOLDS - 1))); do
  fold_tables="$OUTPUT_ROOT/tables/fold-$fold"
  fold_features="$OUTPUT_ROOT/fold-$fold"
  mkdir -p "$fold_features"
  build_features \
    "$fold_tables/train_ordered.csv" \
    "$fold_features/train.npz"
  build_features \
    "$fold_tables/test_ab.csv" \
    "$fold_features/test_ab.npz"
  build_features \
    "$fold_tables/test_ba.csv" \
    "$fold_features/test_ba.npz"
done

echo "Three-API features ready: $OUTPUT_ROOT"
