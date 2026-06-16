#!/usr/bin/env bash
set -euo pipefail

# OSS-based sync helper for environments where GitHub fetch is unstable.
#
# Credentials are intentionally not stored in this repository. Either configure
# ossutil globally, or export:
#   OSS_ACCESS_KEY_ID=...
#   OSS_ACCESS_KEY_SECRET=...
#
# Common commands:
#   bash scripts/oss_sync.sh upload-code
#   bash scripts/oss_sync.sh download-code
#   bash scripts/oss_sync.sh upload-data data/HKU_data_5_total_inbalance.npz
#   bash scripts/oss_sync.sh download-data

OSS_PREFIX="${OSS_PREFIX:-oss://chat-algorithm-data/fg/zfy/mcc-gcn}"
OSS_ENDPOINT="${OSS_ENDPOINT:-https://oss-cn-hangzhou.aliyuncs.com}"
CODE_OBJECT="${CODE_OBJECT:-$OSS_PREFIX/code/MCC-GCN-latest.tar.gz}"
DATA_PREFIX="${DATA_PREFIX:-$OSS_PREFIX/data}"
ARTIFACT_PREFIX="${ARTIFACT_PREFIX:-$OSS_PREFIX/artifacts}"

oss_args=("-e" "$OSS_ENDPOINT")
if [[ -n "${OSS_ACCESS_KEY_ID:-}" && -n "${OSS_ACCESS_KEY_SECRET:-}" ]]; then
  oss_args+=("-i" "$OSS_ACCESS_KEY_ID" "-k" "$OSS_ACCESS_KEY_SECRET")
fi

oss_cp() {
  ossutil cp "$@" "${oss_args[@]}"
}

require_git_root() {
  if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
    echo "error: run this inside the MCC-GCN git checkout" >&2
    exit 1
  fi
}

upload_code() {
  require_git_root
  local root
  root="$(git rev-parse --show-toplevel)"
  local tmp
  tmp="$(mktemp -d)"
  local archive="$tmp/MCC-GCN-latest.tar.gz"
  (
    cd "$root"
    git archive --format=tar.gz --prefix=MCC-GCN/ HEAD > "$archive"
  )
  echo "[upload-code] $(git -C "$root" rev-parse --short HEAD) -> $CODE_OBJECT"
  oss_cp -f "$archive" "$CODE_OBJECT"
  rm -rf "$tmp"
}

download_code() {
  local dest="${1:-$PWD}"
  mkdir -p "$dest"
  local tmp
  tmp="$(mktemp -d)"
  echo "[download-code] $CODE_OBJECT -> $dest"
  oss_cp -f "$CODE_OBJECT" "$tmp/MCC-GCN-latest.tar.gz"
  tar -xzf "$tmp/MCC-GCN-latest.tar.gz" -C "$tmp"
  mkdir -p "$dest/MCC-GCN"
  # rsync keeps data/checkpoints/runs in place unless overwritten by the code package.
  rsync -a --delete \
    --exclude '.git/' \
    --exclude 'data/HKU_data_5_total_inbalance.npz' \
    --exclude 'runs/' \
    "$tmp/MCC-GCN/" "$dest/MCC-GCN/"
  rm -rf "$tmp"
  echo "[download-code] done: $dest/MCC-GCN"
}

upload_data() {
  if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 upload-data PATH [PATH ...]" >&2
    exit 1
  fi
  for path in "$@"; do
    if [[ ! -e "$path" ]]; then
      echo "[skip] missing: $path"
      continue
    fi
    local name
    name="$(basename "$path")"
    echo "[upload-data] $path -> $DATA_PREFIX/$name"
    if [[ -d "$path" ]]; then
      oss_cp -r -f "$path/" "$DATA_PREFIX/$name/"
    else
      oss_cp -f "$path" "$DATA_PREFIX/$name"
    fi
  done
}

download_data() {
  local dest="${1:-data}"
  mkdir -p "$dest"
  echo "[download-data] $DATA_PREFIX/ -> $dest/"
  oss_cp -r -f "$DATA_PREFIX/" "$dest/"
}

upload_artifacts() {
  if [[ "$#" -eq 0 ]]; then
    set -- runs
  fi
  for path in "$@"; do
    if [[ ! -e "$path" ]]; then
      echo "[skip] missing: $path"
      continue
    fi
    local name
    name="$(basename "$path")"
    echo "[upload-artifacts] $path -> $ARTIFACT_PREFIX/$name"
    if [[ -d "$path" ]]; then
      oss_cp -r -f "$path/" "$ARTIFACT_PREFIX/$name/"
    else
      oss_cp -f "$path" "$ARTIFACT_PREFIX/$name"
    fi
  done
}

download_artifacts() {
  local dest="${1:-runs}"
  mkdir -p "$dest"
  echo "[download-artifacts] $ARTIFACT_PREFIX/ -> $dest/"
  oss_cp -r -f "$ARTIFACT_PREFIX/" "$dest/"
}

case "${1:-}" in
  upload-code)
    shift
    upload_code "$@"
    ;;
  download-code)
    shift
    download_code "$@"
    ;;
  upload-data)
    shift
    upload_data "$@"
    ;;
  download-data)
    shift
    download_data "$@"
    ;;
  upload-artifacts)
    shift
    upload_artifacts "$@"
    ;;
  download-artifacts)
    shift
    download_artifacts "$@"
    ;;
  *)
    cat <<EOF
usage: $0 COMMAND [ARGS]

Commands:
  upload-code                 Archive current git HEAD and upload to OSS
  download-code [DEST]        Download code archive and sync into DEST/MCC-GCN
  upload-data PATH [...]      Upload data files/directories to OSS
  download-data [DEST]        Download OSS data prefix into DEST
  upload-artifacts [PATH ...] Upload runs/artifacts to OSS
  download-artifacts [DEST]   Download artifacts into DEST

Environment:
  OSS_PREFIX=$OSS_PREFIX
  OSS_ENDPOINT=$OSS_ENDPOINT
  OSS_ACCESS_KEY_ID=<optional if ossutil is configured>
  OSS_ACCESS_KEY_SECRET=<optional if ossutil is configured>
EOF
    exit 1
    ;;
esac
