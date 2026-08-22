# Pretraining undersampling ablation

This revision experiment isolates the effect of discarding majority-class
source observations during pretraining. It uses the frozen target-API-excluded
source split and the same 170-pair three-API target evaluation as the formal
revision experiment.

## Locked comparison

Both strategies use the same network, optimizer, source validation set, target
set, early-stopping rule, and model seeds 42/43/44.

1. `full-effective-number` retains all 20,076 source training pairs and applies
   effective-number loss weights.
2. `undersampled-uniform` uses one deterministic physical-pair subset and no
   loss weights. The binary task keeps 903 negative and 903 positive pairs. The
   four-class task keeps 866 pairs from each class, or 3,464 pairs in total.

Sampling is performed separately in each task's label space before A/B and B/A
order augmentation. The fixed subset seed is 42 and is shared by all model
seeds, so model-seed variation is not confounded with different sampled data.
The full, imbalanced source validation set and the target set are never sampled.

## Cluster commands

Build the two additional packed-sparse feature files once:

```bash
cd /workspace/MCC-GCN
PYTHON=python SUBSET_SEED=42 \
  bash scripts/prepare_target_api_pretrain_ablation_features.sh
```

Run the complete 2-task, 2-strategy, 3-seed comparison:

```bash
cd /workspace/MCC-GCN
OUTDIR=/workspace/MCC-GCN/runs/revision-three-api-pretrain-ablation-formal \
TENSORBOARD_ROOT=/primus_oss/summary/mcc-gcn/revision-pretrain-ablation \
SEEDS='42 43 44' \
RESUME=1 \
  bash scripts/run_target_api_pretrain_ablation.sh
```

The final table is written to
`pretrain_ablation_summary.md`. The CSV and JSON outputs retain source
validation performance, target accuracy, balanced accuracy, macro F1,
class-specific metrics, per-API metrics, and negative false-positive rate.

This is a controlled revision ablation. It should not be described as an exact
reconstruction of every historical notebook operation.

Formal results and their interpretation are recorded in
`docs/revision-pretrain-undersampling-results.md`.
