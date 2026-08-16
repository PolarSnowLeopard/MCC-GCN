#!/usr/bin/env bash
set -euo pipefail

# Package the frozen three-API CSV tables and manifests for OSS transfer.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT/runs/revision-three-api/data}"
OUTPUT="${OUTPUT:-$ROOT/dist/mcc-gcn-revision-three-api-v2-tables.tar.zst}"

required=(
  "$SOURCE_ROOT/experiment_manifest.json"
  "$SOURCE_ROOT/four_class_target_physical_pairs.csv"
  "$SOURCE_ROOT/pretrain-split/train_ordered_pairs.csv"
  "$SOURCE_ROOT/pretrain-split/validation_ordered_pairs.csv"
  "$SOURCE_ROOT/folds/fold_assignments.csv"
)
for path in "${required[@]}"; do
  test -s "$path" || {
    echo "error: missing frozen experiment data: $path" >&2
    exit 1
  }
done
python3 - "$SOURCE_ROOT/experiment_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
outer = manifest.get("outer_cross_validation", {})
if manifest.get("schema_version") != "mcc-gcn-three-api-experiment-v2":
    raise SystemExit("error: coformer-grouped v2 experiment data is required")
if outer.get("group_by") != "coformer_connectivity_key":
    raise SystemExit("error: outer folds are not grouped by coformer")
if outer.get("coformer_groups_spanning_folds") != 0:
    raise SystemExit("error: a coformer group spans multiple outer folds")
PY
command -v zstd >/dev/null || {
  echo "error: zstd is required" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/data/revision-three-api"
rsync -a --exclude 'features/' "$SOURCE_ROOT/" \
  "$stage/data/revision-three-api/"

tar -C "$stage" -cf - data/revision-three-api \
  | zstd -T0 -19 -f -o "$OUTPUT"

if command -v sha256sum >/dev/null 2>&1; then
  digest="$(sha256sum "$OUTPUT" | awk '{print $1}')"
else
  digest="$(shasum -a 256 "$OUTPUT" | awk '{print $1}')"
fi
printf '%s  %s\n' "$digest" "$(basename "$OUTPUT")" > "$OUTPUT.sha256"
python3 - "$OUTPUT" "$digest" <<'PY'
import json
import sys
from pathlib import Path

archive = Path(sys.argv[1])
metadata = {
    "schema_version": "mcc-gcn-three-api-data-bundle-v2",
    "archive": archive.name,
    "sha256": sys.argv[2],
    "bytes": archive.stat().st_size,
    "contents_root": "data/revision-three-api",
    "requires_ccdc": False,
}
archive.with_suffix(archive.suffix + ".manifest.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(metadata, indent=2, sort_keys=True))
PY

echo "Three-API data bundle ready: $OUTPUT"
