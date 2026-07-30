# Corrected Retraining Workflow

This is the source of truth for the revision experiments. Legacy NPZ files and
notebooks remain frozen evidence only.

## Fixed Experimental Contract

The corrected work produces four primary model artifacts:

| Task | Architecture | Stage | Final evaluation |
| --- | --- | --- | --- |
| Binary (`failed` vs any observed crystal) | historical small GCN | pretrain | all 64 external pairs |
| Binary | historical small GCN | fine-tune on 34 locked pairs | remaining 50 external pairs |
| Four-class | historical large GCN | pretrain | all 64 external pairs |
| Four-class | historical large GCN | fine-tune on 34 locked pairs | remaining 50 external pairs |

The same graph features and physical-pair split are used for both tasks.
Binary labels are derived at load time by mapping class `0` to negative and
classes `1/2/3` to positive.

The binary small model has 23,906 total parameters and only 66 trainable
parameters when the output layer alone is fine-tuned. The four-class large
model has 134,340 total parameters and 260 trainable output-layer parameters.
Every run records these counts so the 34-sample overfitting concern can be
answered directly.

The 34-pair fine-tuning lock is:

- 20 Minoxidil pairs
- 14 fixed external pairs recovered exactly from historical feature hashes

The remaining 50 external pairs are never used for curation, hyperparameter
selection, scheduler decisions, early stopping, or threshold selection.

## Confirmed Corrections

- Original CSD charge/protonation information is preserved.
- Reverse proton-transfer candidates must pass atom-local uniqueness, charge,
  hydrogen, valence, and pair-level checks.
- Ambiguous, multivalent, conflicting, and unparseable rows are quarantined.
- Physical unordered pairs are deduplicated before splitting.
- A/B order augmentation occurs only after the split is frozen.
- No class is downsampled. Effective-number loss weights retain every accepted
  physical pair.
- Padded nodes are removed before PyG graph construction.
- Frozen BatchNorm running statistics remain frozen during fine-tuning.
- Fine-tuning selection uses an internal grouped validation split, then a fresh
  final fit uses all 34 pairs for the selected epoch count.
- Evaluation verifies A/B and B/A row count, label, and pair-key alignment.
- Every training and evaluation directory records configuration, history,
  checkpoint selection, file hashes, and metrics.

## Current Data Counts

The mechanically eligible and conflict-filtered pretraining pool contains
22,669 unique physical pairs:

| Four-class label | Pairs |
| --- | ---: |
| Failed | 1,031 |
| Salt | 10,261 |
| Cocrystal | 10,388 |
| Hydrate or solvate | 989 |

The binary view contains 1,031 negative and 21,638 positive pairs. The frozen
90/10 physical-pair split has zero pair overlap. Both A/B rows remain in the
same split.

## Required Chemistry Review

Fifteen positive Minoxidil rows still require a licensed CCDC re-export. Run:

```bash
python scripts/ccdc/export_manifest.py \
  --manifest data/manifests/minoxidil-15-ccdc-reexport-v1.csv \
  --output data/raw/minoxidil-15.jsonl.gz

python scripts/standardize_csd_export.py \
  data/raw/minoxidil-15.jsonl.gz \
  --output-dir data/interim/minoxidil-15
```

Review the CSD components and fill a copy of
`data/manifests/minoxidil-15-approval-template-v1.csv`. Each row must contain
exactly the two intended model-input neutral reactants and
`review_status=approved`. Do not copy the legacy charge-stripped SMILES.

Materialize the locked task tables:

```bash
python scripts/materialize_locked_datasets.py \
  --finetune-lock data/manifests/finetune-34-lock-v1.csv \
  --external-split-lock data/manifests/external-64-split-v1.csv \
  --approved-minoxidil-pairs data/review/minoxidil-15-approved-v1.csv \
  --output-dir data/corrected-v2/locked
```

The command refuses incomplete approvals, invalid structures, duplicate pairs,
or overlap between the 34-pair fine-tuning set and 50-pair holdout.

## Provisional Work Without CCDC

Lack of a current CCDC license does not block the corrected pretraining pool,
the 64-pair external evaluation, or model/training implementation checks. It
only blocks the final 34-pair fine-tuning dataset and the Minoxidil-only
ablation.

Materialize a strictly separated provisional dataset:

```bash
python scripts/materialize_locked_datasets.py \
  --mode provisional-no-ccdc \
  --finetune-lock data/manifests/finetune-34-lock-v1.csv \
  --external-split-lock data/manifests/external-64-split-v1.csv \
  --output-dir data/corrected-v2/provisional-no-ccdc/locked
```

This excludes all 15 unresolved positive Minoxidil rows. The resulting
fine-tuning set contains 19 physical pairs:

| Four-class label | Pairs |
| --- | ---: |
| Failed | 8 |
| Salt | 5 |
| Cocrystal | 3 |
| Hydrate or solvate | 3 |

The remaining five Minoxidil rows are all failed outcomes, so the provisional
runner does not perform a Minoxidil-only fine-tuning ablation. Every manifest
and run directory records `eligible_for_final_reporting=false`. These results
can validate the implementation and estimate model behavior, but cannot
replace the final 34-pair experiment.

Package the provisional tables for OSS:

```bash
DATASET_PROFILE=provisional-no-ccdc \
  bash scripts/package_corrected_data.sh

bash scripts/oss_sync.sh upload-data \
  dist/mcc-gcn-provisional-no-ccdc-v1-tables.tar.zst \
  dist/mcc-gcn-provisional-no-ccdc-v1-tables.tar.zst.sha256 \
  dist/mcc-gcn-provisional-no-ccdc-v1-tables.tar.zst.manifest.json
```

Restore and smoke-test them on a fresh GPU cluster:

```bash
export DATASET_PROFILE=provisional-no-ccdc
export GIT_REF=fix/data-pipeline-v2
bash scripts/bootstrap_corrected_cluster.sh

PRETRAIN_EPOCHS=1 \
FINETUNE_MAX_EPOCHS=2 \
SEEDS=42 \
EXPERIMENT_PROFILE=provisional-no-ccdc \
OUTDIR=runs/provisional-no-ccdc-smoke \
  bash scripts/run_corrected_experiments.sh
```

## OSS Data Bundle

After chemistry approval:

```bash
CURATION_ROOT=runs/data-curation-v1 \
LOCKED_ROOT=data/corrected-v2/locked \
OUTPUT=dist/mcc-gcn-corrected-v2-tables.tar.zst \
bash scripts/package_corrected_data.sh

bash scripts/oss_sync.sh upload-data \
  dist/mcc-gcn-corrected-v2-tables.tar.zst \
  dist/mcc-gcn-corrected-v2-tables.tar.zst.sha256 \
  dist/mcc-gcn-corrected-v2-tables.tar.zst.manifest.json
```

Only audited CSV/manifests are transferred. Graph NPZ files are rebuilt on the
cluster without CCDC.

## Fresh Cluster Restore

The CUDA image must already provide compatible PyTorch and PyG. Export OSS
credentials in the shell, then:

```bash
export OSS_ACCESS_KEY_ID='...'
export OSS_ACCESS_KEY_SECRET='...'
export GIT_REF=fix/data-pipeline-v2

bash scripts/bootstrap_corrected_cluster.sh
```

The script installs system tools and CCDC-free Python packages, restores the
checksummed data bundle, runs tests, and builds the covalent-only feature set.

## Primary Runs

First run one-epoch end-to-end smoke checks:

```bash
cd /workspace/MCC-GCN
PRETRAIN_EPOCHS=1 \
FINETUNE_MAX_EPOCHS=2 \
SEEDS=42 \
OUTDIR=runs/corrected-smoke \
bash scripts/run_corrected_experiments.sh
```

Then run the locked primary configuration:

```bash
tmux new -s mccgcn
cd /workspace/MCC-GCN

SEEDS='42 43 44' \
PRETRAIN_EPOCHS=400 \
FINETUNE_MAX_EPOCHS=200 \
CLASS_WEIGHTING=effective-number \
EFFECTIVE_NUMBER_BETA=0.9999 \
FINETUNE_LAYERS=1 \
OUTDIR=runs/corrected-primary \
bash scripts/run_corrected_experiments.sh 2>&1 | tee runs/corrected-primary.log
```

The runner also performs the reviewer-requested Minoxidil-only fine-tuning
ablation and evaluates it on all 64 external pairs.

## Locked Ablations

Run these only after the primary smoke succeeds:

1. Class imbalance: `none`, `inverse-frequency`, and `effective-number`.
2. Fine-tuning capacity: last `1`, `2`, or `3` dense layers.
3. Pair edges: covalent-only primary versus `covalent-hbond`.
4. Three random seeds for every result reported as a final comparison.

Do not select the winning ablation on the external 50-pair result. Selection
must use pretraining validation and the internal 34-pair grouped validation
only.
