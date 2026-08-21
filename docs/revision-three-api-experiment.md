# Three-API Revision Experiment

This experiment evaluates Paracetamol, Theophylline, and Riluzole without
requiring a current CCDC installation or license on the GPU cluster.

## Frozen scope

The three APIs are identified by the connectivity block of their InChIKey, so
protonation and component order do not prevent target exclusion.

| API | CAS | Physical pairs | Negative | Salt | Cocrystal | Hydrate/solvate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Paracetamol | 103-90-2 | 42 | 23 | 3 | 14 | 2 |
| Theophylline | 58-55-9 | 110 | 4 | 7 | 81 | 18 |
| Riluzole | 1744-22-5 | 19 | 0 | 7 | 12 | 0 |

One Paracetamol-Theophylline pair belongs to both API subsets. The pooled
benchmark therefore contains 170 unique physical pairs:

| Class | Pairs |
| --- | ---: |
| Negative | 27 |
| Salt | 17 |
| Cocrystal | 106 |
| Hydrate or solvate | 20 |

Six connectivity-level pairs with contradictory source labels are quarantined
and never used for training or evaluation. Riluzole has no available negative
example, so Riluzole-specific binary specificity cannot be estimated from the
current evidence. Binary performance is evaluated on the pooled benchmark.

## Experimental contract

1. Remove every pair containing any of the three APIs from pretraining.
2. Retain the remaining 22,307 curated pretraining pairs without downsampling.
3. Use effective-number class weighting for both pretraining and fine-tuning.
4. Build all experiment features from SMILES with RDKit 2D coordinates and the
   historical covalent-edge, 34-dimensional atom feature configuration.
5. Train the historical small binary MCC-GCN and historical large four-class
   MCC-GCN separately.
6. Evaluate each pretrained model zero-shot on all 170 target pairs.
7. Run five-fold coformer-grouped, label-stratified target-domain fine-tuning.
   Every physical pair is tested exactly once, and the same coformer cannot
   occur in both the training and test partitions of an outer fold.
8. Select the fine-tuning epoch only on an inner pair-grouped validation split,
   then retrain on the complete outer training fold for that epoch count.
9. Average A/B and B/A probabilities before every reported prediction, while
   retaining each orientation's probabilities, label agreement, and
   total-variation discrepancy for the reviewer-requested order analysis.
10. Report pooled and per-API metrics, out-of-fold predictions, confusion
    matrices, and stratified bootstrap confidence intervals.

The outer folds are fixed with seed 42. No target test fold is used for epoch,
class-weight, threshold, or model selection. The 170 pairs form 148 coformer
connectivity groups; no group spans multiple outer test folds.

## Build the frozen data locally

```bash
cd ~/Code/github/MCC-GCN

python scripts/build_target_api_experiment.py \
  --csd-pairs ~/Code/github/HKU-CSD/HKU_data_4_reactions.csv \
  --negative-pairs ~/Code/github/CCDC_Demo/HKU_data_3_real_neg_data.csv \
  --pretrain-pairs runs/data-curation-v2/four_class_physical_pairs.csv \
  --output-dir runs/revision-three-api/data

bash scripts/package_target_api_data.sh
```

The frozen data archive is
`dist/mcc-gcn-revision-three-api-v2-tables.tar.zst`. Its current SHA256 is:

```text
efc5e9fe2f131b20ca22f6b59ca142df5cb9c76d3d1e5cc3809aac7810268b73
```

Upload the code archive and data bundle after committing the implementation:

```bash
bash scripts/oss_sync.sh upload-code
bash scripts/oss_sync.sh upload-data \
  dist/mcc-gcn-revision-three-api-v2-tables.tar.zst \
  dist/mcc-gcn-revision-three-api-v2-tables.tar.zst.sha256 \
  dist/mcc-gcn-revision-three-api-v2-tables.tar.zst.manifest.json
```

## Restore a fresh cluster

Export OSS credentials only in the cluster shell, then run:

```bash
export OSS_ARGS='-e <endpoint> -i <access-key-id> -k <access-key-secret>'
cd /workspace/MCC-GCN
bash scripts/bootstrap_target_api_cluster.sh
```

The bootstrap defaults to OSS for both code and data, installs CCDC-free
dependencies, runs the full unit test suite, and builds packed sparse features.
Set `CODE_SOURCE=github` only when GitHub access is stable.

## Smoke run

```bash
cd /workspace/MCC-GCN

RUN_ID="revision-three-api-smoke-$(date +%Y%m%d-%H%M%S)"
OUTDIR="runs/$RUN_ID" \
TENSORBOARD_ROOT="/primus_oss/summary/mcc-gcn/$RUN_ID" \
TASKS='binary four-class' \
SEEDS=42 \
PRETRAIN_EPOCHS=1 \
FINETUNE_MAX_EPOCHS=2 \
BOOTSTRAP_REPLICATES=100 \
bash scripts/run_target_api_experiment.sh 2>&1 | tee "$RUN_ID.log"
```

## Full run

```bash
cd /workspace/MCC-GCN
tmux new -s mccgcn-revision

RUN_ID="revision-three-api-$(date +%Y%m%d-%H%M%S)"
export OUTDIR="runs/$RUN_ID"
export TENSORBOARD_ROOT="/primus_oss/summary/mcc-gcn/$RUN_ID"

TASKS='binary four-class' \
SEEDS='42 43 44' \
PRETRAIN_EPOCHS=400 \
FINETUNE_MAX_EPOCHS=200 \
FINETUNE_LAYERS=0 \
CLASS_WEIGHTING=effective-number \
bash scripts/run_target_api_experiment.sh 2>&1 | tee "$RUN_ID.log"
```

For each task and seed, the runner writes:

- the target-excluded pretraining checkpoint and zero-shot summary;
- five selection runs and five final-fit checkpoints;
- one prediction file per held-out fold;
- a 170-pair out-of-fold prediction table;
- pooled and per-API metrics with bootstrap confidence intervals.

TensorBoard events are written below the configured summary root by task,
seed, stage, and fold.
