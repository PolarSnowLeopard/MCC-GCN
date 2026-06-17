#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh training cluster from OSS.
#
# This script prepares the environment and data only. It does not change
# training semantics or inject any data balancing logic.

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/MCC-GCN}"
OSS_PREFIX="${OSS_PREFIX:-oss://chat-algorithm-data/fg/zfy/mcc-gcn}"
OSS_ENDPOINT="${OSS_ENDPOINT:-https://oss-cn-hangzhou.aliyuncs.com}"
CODE_OBJECT="${CODE_OBJECT:-$OSS_PREFIX/code/MCC-GCN-latest.tar.gz}"
PRETRAIN_DATA_NAME="${PRETRAIN_DATA_NAME:-HKU_data_5_total_inbalance.npz}"
PRETRAIN_DATA_OBJECT="${PRETRAIN_DATA_OBJECT:-$OSS_PREFIX/data/$PRETRAIN_DATA_NAME.zst}"
PRETRAIN_ZST_SHA256="${PRETRAIN_ZST_SHA256:-7985bf68eb5f9c405fee5d6031e004304d6b815dd0ae9a05fda9535f6dd2bcdc}"
PRETRAIN_NPZ_BYTES="${PRETRAIN_NPZ_BYTES:-36400335778}"
INSTALL_SYSTEM_PACKAGES="${INSTALL_SYSTEM_PACKAGES:-1}"
INSTALL_PY_DEPS="${INSTALL_PY_DEPS:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
RUN_PADDING_CHECK="${RUN_PADDING_CHECK:-1}"
RUN_RETRAIN_SMOKE="${RUN_RETRAIN_SMOKE:-0}"
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

oss_args=("-e" "$OSS_ENDPOINT")
if [[ -n "${OSS_ACCESS_KEY_ID:-}" && -n "${OSS_ACCESS_KEY_SECRET:-}" ]]; then
  oss_args+=("-i" "$OSS_ACCESS_KEY_ID" "-k" "$OSS_ACCESS_KEY_SECRET")
fi

oss_cp() {
  ossutil cp "$@" "${oss_args[@]}"
}

file_size() {
  stat -c '%s' "$1"
}

install_system_packages() {
  [[ "$INSTALL_SYSTEM_PACKAGES" == "1" ]] || return 0
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing system tools"
    export DEBIAN_FRONTEND=noninteractive
    as_root apt-get update
    as_root apt-get install -y ca-certificates curl unzip git rsync tmux zstd
  else
    log "apt-get not found; assuming system tools are already available"
  fi
}

install_ossutil() {
  if command -v ossutil >/dev/null 2>&1; then
    log "ossutil found: $(command -v ossutil)"
    return 0
  fi
  log "Installing ossutil"
  local tmp
  tmp="$(mktemp -d)"
  curl -fsSL \
    https://gosspublic.alicdn.com/ossutil/1.7.19/ossutil-v1.7.19-linux-amd64.zip \
    -o "$tmp/ossutil.zip"
  unzip -q "$tmp/ossutil.zip" -d "$tmp"
  as_root install -m 0755 "$tmp/ossutil-v1.7.19-linux-amd64/ossutil" /usr/local/bin/ossutil
  rm -rf "$tmp"
  ossutil version
}

download_code() {
  log "Downloading code archive"
  mkdir -p "$WORKSPACE"
  local tmp
  tmp="$(mktemp -d)"
  oss_cp -f "$CODE_OBJECT" "$tmp/MCC-GCN-latest.tar.gz"
  tar -xzf "$tmp/MCC-GCN-latest.tar.gz" -C "$tmp"
  mkdir -p "$REPO_DIR"
  rsync -a --delete \
    --exclude '.git/' \
    --exclude 'data/HKU_data_5_total_inbalance.npz' \
    --exclude 'data/HKU_data_5_total_inbalance.npz.zst' \
    --exclude 'runs/' \
    "$tmp/MCC-GCN/" "$REPO_DIR/"
  rm -rf "$tmp"
  log "Code ready: $REPO_DIR"
}

install_python_deps() {
  [[ "$INSTALL_PY_DEPS" == "1" ]] || return 0
  log "Installing Python dependencies that are not CUDA-version-sensitive"
  python -m pip install \
    -i "$PIP_INDEX_URL" \
    --extra-index-url "$PIP_EXTRA_INDEX_URL" \
    numpy pandas scipy scikit-learn tqdm requests cirpy rdkit
  (
    cd "$REPO_DIR"
    python -m pip install -e . --no-deps
  )
}

verify_python_stack() {
  log "Verifying Python/CUDA stack"
  python - <<'PY'
import torch
import torch_geometric
from torch_geometric.nn import GCNConv
from rdkit import Chem
import numpy, pandas, scipy, sklearn, tqdm, requests, cirpy

print("torch:", torch.__version__)
print("cuda compiled:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("pyg:", torch_geometric.__version__)
print("rdkit ok:", Chem.MolFromSmiles("CCO") is not None)
print("GCNConv:", GCNConv)
PY
  if [[ "$REQUIRE_CUDA" == "1" ]]; then
    python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else "CUDA is not available")
PY
  fi
}

download_pretrain_data() {
  log "Preparing pretraining NPZ"
  mkdir -p "$REPO_DIR/data"
  local target="$REPO_DIR/data/$PRETRAIN_DATA_NAME"
  local compressed="$target.zst"

  if [[ -s "$target" ]] && [[ "$(file_size "$target")" == "$PRETRAIN_NPZ_BYTES" ]]; then
    log "Pretraining NPZ already exists with expected size: $target"
    return 0
  fi

  rm -f "$target" "$target.tmp"
  if [[ ! -s "$compressed" ]]; then
    log "Downloading compressed pretraining data"
    oss_cp -f \
      "$PRETRAIN_DATA_OBJECT" \
      "$compressed" \
      --bigfile-threshold 1048576 \
      --part-size 1048576 \
      --parallel 16
  fi

  log "Validating compressed pretraining data"
  echo "$PRETRAIN_ZST_SHA256  $compressed" | sha256sum -c -
  zstd -t "$compressed"

  log "Decompressing pretraining data"
  zstd -d -f -c "$compressed" > "$target.tmp"
  mv "$target.tmp" "$target"
  local actual_size
  actual_size="$(file_size "$target")"
  if [[ "$actual_size" != "$PRETRAIN_NPZ_BYTES" ]]; then
    echo "error: expected $PRETRAIN_NPZ_BYTES bytes, got $actual_size for $target" >&2
    exit 1
  fi
  ls -lh "$target" "$compressed"
}

verify_repo_data() {
  log "Verifying repository data files"
  (
    cd "$REPO_DIR"
    for path in \
      data/HKU_data_5_total_inbalance.npz \
      data/HKU_data_6_FT_minoxidil_balanced_with_exp.npz \
      data/HKU_data_6_experiment.npz \
      data/HKU_data_6_experiment_1.npz \
      data/HKU_data_6_experiment_2.npz \
      checkpoints/best_FT_model.pth
    do
      test -s "$path" || { echo "missing or empty: $path" >&2; exit 1; }
      ls -lh "$path"
    done
  )
}

run_padding_check() {
  [[ "$RUN_PADDING_CHECK" == "1" ]] || return 0
  log "Running padding invariance check"
  (
    cd "$REPO_DIR"
    python scripts/check_padding_invariance.py \
      --data data/HKU_data_6_experiment_1.npz \
      --model checkpoints/best_FT_model.pth
  )
}

run_retrain_smoke() {
  [[ "$RUN_RETRAIN_SMOKE" == "1" ]] || return 0
  log "Running 1-epoch retraining smoke test"
  (
    cd "$REPO_DIR"
    OUTDIR="runs/smoke-bootstrap-$(date +%Y%m%d-%H%M%S)" \
    PRETRAIN_EPOCHS=1 \
    FT_EPOCHS=1 \
    PRETRAIN_BATCH_SIZE=64 \
    FT_BATCH_SIZE=16 \
    bash scripts/retrain_from_npz.sh
  )
}

main() {
  install_system_packages
  install_ossutil
  download_code
  install_python_deps
  verify_python_stack
  download_pretrain_data
  verify_repo_data
  run_padding_check
  run_retrain_smoke
  log "Bootstrap complete: $REPO_DIR"
}

main "$@"
