#!/usr/bin/env bash
set -euo pipefail

# Package only audited tables and locked evaluation data. Large graph features
# are rebuilt on the GPU cluster without CCDC.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURATION_ROOT="${CURATION_ROOT:-$ROOT/runs/data-curation-v2}"
DATASET_PROFILE="${DATASET_PROFILE:-final}"

case "$DATASET_PROFILE" in
  final)
    LOCKED_ROOT="${LOCKED_ROOT:-$ROOT/data/corrected-v2/locked}"
    OUTPUT="${OUTPUT:-$ROOT/dist/mcc-gcn-corrected-v2-tables.tar.zst}"
    locked_stage_path="data/corrected-v2/locked"
    finetune_table="four_class_finetune_34_physical_pairs.csv"
    minoxidil_table="four_class_minoxidil_20_physical_pairs.csv"
    eligible_for_final_reporting=true
    ;;
  provisional-no-ccdc)
    LOCKED_ROOT="${LOCKED_ROOT:-$ROOT/data/corrected-v2/provisional-no-ccdc/locked}"
    OUTPUT="${OUTPUT:-$ROOT/dist/mcc-gcn-provisional-no-ccdc-v2-tables.tar.zst}"
    locked_stage_path="data/corrected-v2/provisional-no-ccdc/locked"
    finetune_table="four_class_provisional_finetune_19_physical_pairs.csv"
    minoxidil_table="four_class_provisional_minoxidil_5_physical_pairs.csv"
    eligible_for_final_reporting=false
    ;;
  *)
    echo "error: unknown DATASET_PROFILE=$DATASET_PROFILE" >&2
    exit 1
    ;;
esac

required=(
  "$CURATION_ROOT/curation_manifest.json"
  "$CURATION_ROOT/four_class_physical_pairs.csv"
  "$CURATION_ROOT/four-class-split/train_ordered_pairs.csv"
  "$CURATION_ROOT/four-class-split/validation_ordered_pairs.csv"
  "$CURATION_ROOT/four-class-split/split_manifest.csv"
  "$LOCKED_ROOT/locked_dataset_manifest.json"
  "$LOCKED_ROOT/$finetune_table"
  "$LOCKED_ROOT/$minoxidil_table"
  "$LOCKED_ROOT/four_class_external_64_physical_pairs.csv"
  "$LOCKED_ROOT/four_class_holdout_50_physical_pairs.csv"
)
for path in "${required[@]}"; do
  test -s "$path" || {
    echo "error: missing or empty corrected data: $path" >&2
    exit 1
  }
done
command -v zstd >/dev/null || {
  echo "error: zstd is required" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p \
  "$stage/data/corrected-v2/curation" \
  "$stage/$locked_stage_path"
rsync -a "$CURATION_ROOT/" "$stage/data/corrected-v2/curation/"
rsync -a "$LOCKED_ROOT/" "$stage/$locked_stage_path/"

tar -C "$stage" -cf - data/corrected-v2 | zstd -T0 -19 -f -o "$OUTPUT"

if command -v sha256sum >/dev/null 2>&1; then
  digest="$(sha256sum "$OUTPUT" | awk '{print $1}')"
else
  digest="$(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
fi
printf '%s  %s\n' "$digest" "$(basename "$OUTPUT")" > "$OUTPUT.sha256"
python3 - "$OUTPUT" "$digest" "$DATASET_PROFILE" \
  "$eligible_for_final_reporting" "$locked_stage_path" <<'PY'
import json
import os
import sys
from pathlib import Path

archive = Path(sys.argv[1])
metadata = {
    "schema_version": "mcc-gcn-corrected-data-bundle-v2",
    "archive": archive.name,
    "sha256": sys.argv[2],
    "bytes": archive.stat().st_size,
    "contents_root": "data/corrected-v2",
    "dataset_profile": sys.argv[3],
    "eligible_for_final_reporting": sys.argv[4] == "true",
    "locked_tables_root": sys.argv[5],
}
archive.with_suffix(archive.suffix + ".manifest.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(metadata, indent=2, sort_keys=True))
PY

echo "Corrected data bundle ready: $OUTPUT (profile=$DATASET_PROFILE)"
