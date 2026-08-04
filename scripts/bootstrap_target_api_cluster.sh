#!/usr/bin/env bash
set -euo pipefail

# Restore a fresh GPU cluster from OSS and prepare the three-API experiment.

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/MCC-GCN}"
CODE_SOURCE="${CODE_SOURCE:-oss}"
GIT_URL="${GIT_URL:-https://github.com/PolarSnowLeopard/MCC-GCN.git}"
GIT_REF="${GIT_REF:-fix/data-pipeline-v2}"
OSS_PREFIX="${OSS_PREFIX:-oss://chat-algorithm-data/fg/zfy/mcc-gcn}"
OSS_ENDPOINT="${OSS_ENDPOINT:-https://oss-cn-hangzhou.aliyuncs.com}"
CODE_OBJECT="${CODE_OBJECT:-$OSS_PREFIX/code/MCC-GCN-latest.tar.gz}"
DATA_OBJECT="${DATA_OBJECT:-$OSS_PREFIX/data/mcc-gcn-revision-three-api-v1-tables.tar.zst}"
DATA_SHA256_OBJECT="${DATA_SHA256_OBJECT:-$DATA_OBJECT.sha256}"
BUILD_FEATURES="${BUILD_FEATURES:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
SKIP_CODE_UPDATE="${SKIP_CODE_UPDATE:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://pypi.ngc.nvidia.com}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "error: root or sudo is required for: $*" >&2
    exit 1
  fi
}

if [[ -n "${OSS_ARGS:-}" ]]; then
  IFS=' ' read -r -a oss_args <<< "$OSS_ARGS"
else
  oss_args=("-e" "$OSS_ENDPOINT")
  if [[ -n "${OSS_ACCESS_KEY_ID:-}" && -n "${OSS_ACCESS_KEY_SECRET:-}" ]]; then
    oss_args+=("-i" "$OSS_ACCESS_KEY_ID" "-k" "$OSS_ACCESS_KEY_SECRET")
  fi
fi

oss_cp() {
  ossutil cp "$@" "${oss_args[@]}"
}

install_tools() {
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    as_root apt-get update
    as_root apt-get install -y \
      ca-certificates curl git rsync tmux unzip zstd
  fi
  if ! command -v ossutil >/dev/null 2>&1; then
    local temp
    temp="$(mktemp -d)"
    curl -fsSL \
      https://gosspublic.alicdn.com/ossutil/1.7.19/ossutil-v1.7.19-linux-amd64.zip \
      -o "$temp/ossutil.zip"
    unzip -q "$temp/ossutil.zip" -d "$temp"
    as_root install -m 0755 \
      "$temp/ossutil-v1.7.19-linux-amd64/ossutil" \
      /usr/local/bin/ossutil
    rm -rf "$temp"
  fi
}

restore_code_from_oss() {
  local temp
  temp="$(mktemp -d)"
  oss_cp -f "$CODE_OBJECT" "$temp/MCC-GCN-latest.tar.gz"
  tar -xzf "$temp/MCC-GCN-latest.tar.gz" -C "$temp"
  mkdir -p "$REPO_DIR"
  rsync -a --delete \
    --exclude '.git/' \
    --exclude 'data/' \
    --exclude 'runs/' \
    "$temp/MCC-GCN/" "$REPO_DIR/"
  rm -rf "$temp"
}

restore_code_from_github() {
  mkdir -p "$WORKSPACE"
  if [[ -d "$REPO_DIR/.git" ]]; then
    git -C "$REPO_DIR" -c http.version=HTTP/1.1 fetch origin "$GIT_REF"
    git -C "$REPO_DIR" checkout -B "$GIT_REF" FETCH_HEAD
  else
    git -c http.version=HTTP/1.1 clone \
      --branch "$GIT_REF" --single-branch "$GIT_URL" "$REPO_DIR"
  fi
}

restore_code() {
  if [[ "$SKIP_CODE_UPDATE" == "1" ]]; then
    test -s "$REPO_DIR/scripts/run_target_api_experiment.sh" || {
      echo "error: target experiment code is missing from $REPO_DIR" >&2
      exit 1
    }
    log "Skipping code update"
    return
  fi
  log "Restoring code from $CODE_SOURCE"
  case "$CODE_SOURCE" in
    oss)
      restore_code_from_oss
      ;;
    github)
      restore_code_from_github
      ;;
    *)
      echo "error: CODE_SOURCE must be oss or github" >&2
      exit 1
      ;;
  esac
}

install_python_deps() {
  log "Installing CCDC-free Python dependencies"
  python -m pip install \
    -i "$PIP_INDEX_URL" \
    --extra-index-url "$PIP_EXTRA_INDEX_URL" \
    numpy pandas scipy scikit-learn tqdm requests cirpy \
    'rdkit==2025.9.2' torch-geometric tensorboard
  python -m pip install -e "$REPO_DIR" --no-deps
  python - <<'PY'
import torch
import torch_geometric
from rdkit import Chem

print("torch:", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("rdkit:", Chem.MolFromSmiles("CCO") is not None)
raise SystemExit(0 if torch.cuda.is_available() else "CUDA is unavailable")
PY
}

restore_data() {
  log "Downloading frozen three-API data bundle"
  local temp archive checksum
  temp="$(mktemp -d)"
  archive="$temp/$(basename "$DATA_OBJECT")"
  checksum="$archive.sha256"
  oss_cp -f "$DATA_OBJECT" "$archive"
  oss_cp -f "$DATA_SHA256_OBJECT" "$checksum"
  (
    cd "$temp"
    sha256sum -c "$(basename "$checksum")"
  )
  zstd -t "$archive"
  zstd -d -c "$archive" | tar -xf - -C "$REPO_DIR"
  test -s "$REPO_DIR/data/revision-three-api/experiment_manifest.json"
  rm -rf "$temp"
}

main() {
  install_tools
  restore_code
  install_python_deps
  restore_data
  if [[ "$RUN_TESTS" == "1" ]]; then
    (
      cd "$REPO_DIR"
      python -m unittest discover -s tests -v
    )
  fi
  if [[ "$BUILD_FEATURES" == "1" ]]; then
    (
      cd "$REPO_DIR"
      bash scripts/prepare_target_api_features.sh
    )
  fi
  log "Three-API cluster restore complete: $REPO_DIR"
}

main "$@"
