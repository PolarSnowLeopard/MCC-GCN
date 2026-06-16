# OSS Sync

Use this path when the training cluster cannot reliably fetch from GitHub.

The helper script is `scripts/oss_sync.sh`. It follows the same pattern used in
the existing `med_rl` and `MemLoRA` projects: `ossutil cp` copies a prepared code
or data tree between OSS and the cluster.

## Credentials

Do not commit access keys. Either configure `ossutil`, or export:

```bash
export OSS_ACCESS_KEY_ID=...
export OSS_ACCESS_KEY_SECRET=...
export OSS_ENDPOINT=https://oss-cn-hangzhou.aliyuncs.com
```

Default prefix:

```bash
export OSS_PREFIX=oss://chat-algorithm-data/fg/zfy/mcc-gcn
```

## Upload Code From Local Machine

From the local MCC-GCN checkout:

```bash
bash scripts/oss_sync.sh upload-code
```

This uploads a clean `git archive` of the current `HEAD` to:

```text
$OSS_PREFIX/code/MCC-GCN-latest.tar.gz
```

## Download Code On Cluster

From a parent directory such as `/workspace`:

```bash
bash /workspace/MCC-GCN/scripts/oss_sync.sh download-code /workspace
cd /workspace/MCC-GCN
python -m pip install -e . --no-deps
```

If the existing repository does not have the helper yet, download directly:

```bash
ossutil cp -f \
  oss://chat-algorithm-data/fg/zfy/mcc-gcn/code/MCC-GCN-latest.tar.gz \
  /tmp/MCC-GCN-latest.tar.gz \
  -e https://oss-cn-hangzhou.aliyuncs.com

mkdir -p /workspace
tar -xzf /tmp/MCC-GCN-latest.tar.gz -C /workspace
cd /workspace/MCC-GCN
python -m pip install -e . --no-deps
```

Add `-i ... -k ...` to the `ossutil` command if the cluster has no global
ossutil config.

## Sync Data

The large pretraining NPZ is stored in OSS as a compressed object to avoid
transferring the 34GB padded file directly:

```text
oss://chat-algorithm-data/fg/zfy/mcc-gcn/data/HKU_data_5_total_inbalance.npz.zst
```

Download and decompress it on the cluster:

```bash
cd /workspace/MCC-GCN
bash scripts/oss_sync.sh download-pretrain-data data
ls -lh data/HKU_data_5_total_inbalance.npz
```

Upload selected data files:

```bash
bash scripts/oss_sync.sh upload-data \
  data/HKU_data_5_total_inbalance.npz \
  data/HKU_data_6_FT_minoxidil_balanced_with_exp.npz \
  data/HKU_data_6_experiment.npz \
  data/HKU_data_6_experiment_1.npz \
  data/HKU_data_6_experiment_2.npz
```

Download data on cluster:

```bash
bash scripts/oss_sync.sh download-data data
```

## Sync Training Outputs

Backup:

```bash
bash scripts/oss_sync.sh upload-artifacts runs/a800-fixed-loader-001
```

Restore:

```bash
bash scripts/oss_sync.sh download-artifacts runs
```
