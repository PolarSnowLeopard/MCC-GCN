#!/usr/bin/env bash
set -euo pipefail

# Build one shared set of graph features. Binary labels are derived at load
# time, so the expensive V/A arrays are not duplicated.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURATION_ROOT="${CURATION_ROOT:-$ROOT/data/corrected-v2/curation}"
DATASET_PROFILE="${DATASET_PROFILE:-final}"
FEATURE_VARIANT="${FEATURE_VARIANT:-covalent}"
FORCE="${FORCE:-0}"

case "$DATASET_PROFILE" in
  final)
    LOCKED_ROOT="${LOCKED_ROOT:-$ROOT/data/corrected-v2/locked}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/data/corrected-v2/features/$FEATURE_VARIANT}"
    finetune_table="four_class_finetune_34_physical_pairs.csv"
    minoxidil_table="four_class_minoxidil_20_physical_pairs.csv"
    finetune_name="finetune_34"
    minoxidil_name="minoxidil_20"
    ;;
  provisional-no-ccdc)
    LOCKED_ROOT="${LOCKED_ROOT:-$ROOT/data/corrected-v2/provisional-no-ccdc/locked}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/data/corrected-v2/provisional-no-ccdc/features/$FEATURE_VARIANT}"
    finetune_table="four_class_provisional_finetune_19_physical_pairs.csv"
    minoxidil_table="four_class_provisional_minoxidil_5_physical_pairs.csv"
    finetune_name="finetune_19_provisional"
    minoxidil_name="minoxidil_5_provisional"
    ;;
  *)
    echo "error: unknown DATASET_PROFILE=$DATASET_PROFILE" >&2
    exit 1
    ;;
esac

case "$FEATURE_VARIANT" in
  covalent)
    feature_args=()
    ;;
  covalent-hbond)
    feature_args=(--hbond)
    ;;
  covalent-hbond-pipi-contact)
    feature_args=(--hbond --pipi-stack --contact)
    ;;
  *)
    echo "error: unknown FEATURE_VARIANT=$FEATURE_VARIANT" >&2
    exit 1
    ;;
esac

required=(
  "$CURATION_ROOT/four-class-split/train_ordered_pairs.csv"
  "$CURATION_ROOT/four-class-split/validation_ordered_pairs.csv"
  "$LOCKED_ROOT/$finetune_table"
  "$LOCKED_ROOT/$minoxidil_table"
  "$LOCKED_ROOT/four_class_external_64_physical_pairs.csv"
  "$LOCKED_ROOT/four_class_holdout_50_physical_pairs.csv"
)
for path in "${required[@]}"; do
  test -s "$path" || {
    echo "error: missing or empty input: $path" >&2
    exit 1
  }
done

mkdir -p "$OUTPUT_ROOT/tables"

python "$ROOT/scripts/augment_pair_orders.py" \
  --input "$LOCKED_ROOT/$finetune_table" \
  --output "$OUTPUT_ROOT/tables/${finetune_name}_ordered.csv"
python "$ROOT/scripts/augment_pair_orders.py" \
  --input "$LOCKED_ROOT/$minoxidil_table" \
  --output "$OUTPUT_ROOT/tables/${minoxidil_name}_ordered.csv"
python "$ROOT/scripts/augment_pair_orders.py" \
  --input "$LOCKED_ROOT/four_class_external_64_physical_pairs.csv" \
  --forward-output "$OUTPUT_ROOT/tables/external_64_ab.csv" \
  --reverse-output "$OUTPUT_ROOT/tables/external_64_ba.csv"
python "$ROOT/scripts/augment_pair_orders.py" \
  --input "$LOCKED_ROOT/four_class_holdout_50_physical_pairs.csv" \
  --forward-output "$OUTPUT_ROOT/tables/holdout_50_ab.csv" \
  --reverse-output "$OUTPUT_ROOT/tables/holdout_50_ba.csv"

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
    --adjacency-type OnlyCovalentBond \
    "${feature_args[@]}"
}

build_features \
  "$CURATION_ROOT/four-class-split/train_ordered_pairs.csv" \
  "$OUTPUT_ROOT/pretrain_train.npz"
build_features \
  "$CURATION_ROOT/four-class-split/validation_ordered_pairs.csv" \
  "$OUTPUT_ROOT/pretrain_validation.npz"
build_features \
  "$OUTPUT_ROOT/tables/${finetune_name}_ordered.csv" \
  "$OUTPUT_ROOT/${finetune_name}.npz"
build_features \
  "$OUTPUT_ROOT/tables/${minoxidil_name}_ordered.csv" \
  "$OUTPUT_ROOT/${minoxidil_name}.npz"
build_features \
  "$OUTPUT_ROOT/tables/external_64_ab.csv" \
  "$OUTPUT_ROOT/external_64_ab.npz"
build_features \
  "$OUTPUT_ROOT/tables/external_64_ba.csv" \
  "$OUTPUT_ROOT/external_64_ba.npz"
build_features \
  "$OUTPUT_ROOT/tables/holdout_50_ab.csv" \
  "$OUTPUT_ROOT/holdout_50_ab.npz"
build_features \
  "$OUTPUT_ROOT/tables/holdout_50_ba.csv" \
  "$OUTPUT_ROOT/holdout_50_ba.npz"

echo "Corrected features ready: $OUTPUT_ROOT (profile=$DATASET_PROFILE)"
