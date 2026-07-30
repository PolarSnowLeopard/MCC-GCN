# Cluster Bootstrap

For corrected revision experiments, use
`scripts/bootstrap_corrected_cluster.sh` and
`docs/corrected-retraining.md`. The workflow below restores the frozen
legacy-NPZ environment only.

Use this when a fresh training cluster starts without code, data, or local
project installation. This path does not require CCDC.

The bootstrap script restores:

- system tools: `zstd`, `rsync`, `git`, `tmux`, `curl`, `unzip`
- `ossutil` if missing
- the latest OSS code archive
- Python package installation for this repository
- the compressed pretraining NPZ from OSS, verified by SHA256 and decompressed
- a CUDA/RDKit/PyG environment check
- the padding invariance smoke check

It does not change training semantics or implement any data sampling logic.

## Start From A Fresh Cluster

Export OSS credentials in the shell. Do not paste keys into logs or chat.

```bash
export OSS_ACCESS_KEY_ID='...'
export OSS_ACCESS_KEY_SECRET='...'
export OSS_ENDPOINT=https://oss-cn-hangzhou.aliyuncs.com
export OSS_PREFIX=oss://chat-algorithm-data/fg/zfy/mcc-gcn
```

If `ossutil` already exists:

```bash
ossutil cp -f \
  "$OSS_PREFIX/code/bootstrap_cluster.sh" \
  /tmp/bootstrap_cluster.sh \
  -e "$OSS_ENDPOINT" \
  -i "$OSS_ACCESS_KEY_ID" \
  -k "$OSS_ACCESS_KEY_SECRET"

bash /tmp/bootstrap_cluster.sh
```

If `ossutil` is missing, install it first:

```bash
apt-get update
apt-get install -y ca-certificates curl unzip
curl -fsSL \
  https://gosspublic.alicdn.com/ossutil/1.7.19/ossutil-v1.7.19-linux-amd64.zip \
  -o /tmp/ossutil.zip
unzip -q /tmp/ossutil.zip -d /tmp
install -m 0755 /tmp/ossutil-v1.7.19-linux-amd64/ossutil /usr/local/bin/ossutil
```

Then run the `ossutil cp` command above.

## Optional 1-Epoch Smoke

By default the bootstrap only prepares the cluster and runs the padding
invariance check. To also run the full pretrain -> fine-tune -> evaluate
pipeline for one epoch:

```bash
RUN_RETRAIN_SMOKE=1 bash /tmp/bootstrap_cluster.sh
```

## Training

After bootstrap:

```bash
tmux new -s mccgcn
cd /workspace/MCC-GCN

OUTDIR=runs/a800-fixed-loader-$(date +%Y%m%d-%H%M%S) \
PRETRAIN_EPOCHS=400 \
FT_EPOCHS=50 \
PRETRAIN_BATCH_SIZE=64 \
FT_BATCH_SIZE=16 \
bash scripts/retrain_from_npz.sh
```

Detach from tmux with `Ctrl-b`, then `d`. Reattach:

```bash
tmux a -t mccgcn
```

## Notes

- The large pretraining data object is
  `oss://chat-algorithm-data/fg/zfy/mcc-gcn/data/HKU_data_5_total_inbalance.npz.zst`.
- Its SHA256 is
  `7985bf68eb5f9c405fee5d6031e004304d6b815dd0ae9a05fda9535f6dd2bcdc`.
- The decompressed NPZ size must be `36400335778` bytes.
