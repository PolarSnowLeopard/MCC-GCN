#!/usr/bin/env bash
set -euo pipefail

VENV=${DEEPCOCRYSTAL_VENV:-/workspace/venvs/deepcocrystal-tf218}
REPO=${DEEPCOCRYSTAL_REPO:-/workspace/deep-cocrystal}
PYTHON=${PYTHON:-python3}
COMMIT=2805197eff579844b6a45f71a67e2638ede7a14e

retry() {
  local attempt=1
  local maximum=5
  until "$@"; do
    if ((attempt >= maximum)); then
      return 1
    fi
    sleep $((attempt * 3))
    attempt=$((attempt + 1))
  done
}

mkdir -p "$(dirname "$VENV")"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install \
  'tensorflow[and-cuda]==2.18.1' \
  'tf-keras==2.18.0' \
  'numpy==1.26.4' \
  'pandas==2.2.3' \
  'scikit-learn==1.5.2' \
  'rdkit==2025.9.2'

if [[ ! -d "$REPO/.git" ]]; then
  retry git -c http.version=HTTP/1.1 clone \
    https://github.com/molML/deep-cocrystal.git "$REPO"
fi

if ! git -C "$REPO" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
  retry git -C "$REPO" -c http.version=HTTP/1.1 fetch origin "$COMMIT"
fi
git -C "$REPO" checkout --detach "$COMMIT"

(
  cd "$REPO"
  export TF_USE_LEGACY_KERAS=1
  "$VENV/bin/python" - <<'PY'
import tensorflow as tf

from deepcocrystal.deepcocrystal import DeepCocrystal

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("TensorFlow cannot see a GPU")

model = DeepCocrystal()
output = model(
    [tf.constant(["C C O"]), tf.constant(["O"])],
    training=False,
)
if tuple(output.shape) != (1, 1):
    raise RuntimeError(f"Unexpected model output shape: {output.shape}")

print("tensorflow", tf.__version__)
print("gpu", gpus[0].name)
print("deepcocrystal_commit", "2805197eff579844b6a45f71a67e2638ede7a14e")
print("forward_shape", tuple(output.shape))
PY
)

printf 'DeepCocrystal environment ready: %s\n' "$VENV"
