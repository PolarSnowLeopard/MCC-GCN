# Legacy Baseline Freeze

This document freezes the evidence used by the submitted MCC-GCN paper before
corrected data rebuilding and retraining begin. It records historical behavior;
it does not approve the legacy data or methodology.

The machine-readable inventory is
`data/manifests/legacy-baseline-v1.json`.

## Immutable Sources

| Source | Commit | Purpose |
| --- | --- | --- |
| `PolarSnowLeopard/MCC-GCN` | `d5d599b1cbbcb92d28dbbfb2435a7ab6ae26feeb` | Public paper baseline |
| `PolarSnowLeopard/MCC-GCN` | `40951089ba698ea4ba9000f9eacc5e3cdb8f9d16` | Full historical data-build snapshot |
| `PolarSnowLeopard/HKU-CSD` | `b8ed6db` | Original CSD extraction assets |
| `PolarSnowLeopard/CCDC_Demo` | `f5d4ba0` | Executed experiment notebooks |
| Private Overleaf repository | `e3c756f9081c89a8c99da8b87d5a7565683ece0e` | Submitted manuscript source |

Corrected work must not rewrite these commits, checkpoints, CSV files, NPZ
files, or notebook outputs. New artifacts must use versioned paths and include
their own manifests.

## Model And Evaluation Boundary

- The fine-tuning set contains 34 physical molecular pairs: 20 minoxidil pairs
  and 14 pairs selected from the 64-pair external experiment.
- A/B reversal expands those 34 physical pairs to 68 ordered feature rows.
- The final external holdout contains the remaining 50 physical pairs.
- `HKU_data_6_experiment_1.npz` and
  `HKU_data_6_experiment_2.npz` are the forward and reverse forms of the same
  50 pairs.
- `HKU_data_6_experiment.npz` contains both orders, so its 100 rows are not 100
  independent experiments.
- The holdout must not be used for hyperparameter selection, early stopping,
  threshold selection, or data-curation decisions in corrected experiments.

The historical four-class checkpoint produced 0.5800 overall accuracy with the
legacy padded-node loader. That result remains the paper baseline. Running the
same checkpoint through the fixed loader is a compatibility diagnostic, not a
replacement historical result.

## Pretraining Evidence

The executed four-class notebook `CCDC_Demo/5. Models.ipynb` loads
`HKU_data_5_total_inbalance.npz`. Its saved output reports 68,433 rows padded to
178 nodes and a random 90/10 row-level train/validation split.

The binary notebook and dataset scripts are not internally consistent:

- `Scrpts/data_5_total_inbalance_bin.py` creates both an imbalanced pool and a
  2,104-per-class balanced selection.
- The binary notebook source names the imbalanced file, while its saved tensor
  shape is consistent with the balanced selection after failed conversions.
- Several notebook sources were edited after their displayed outputs were
  produced.

The corrected workflow must therefore rebuild both tasks from explicit pair
manifests. Notebook filenames or stale cell outputs are not sufficient
provenance.

## Preserved Raw Chemistry

The original `HKU-CSD/HKU_data_4_reactions.csv` still contains charged CSD
component SMILES:

- 34,162 CSD rows
- 21,494 rows with at least one nonzero component charge
- 21,529 rows labeled as salt
- 930 rows that current RDKit cannot sanitize directly
- 194 unordered molecular pairs with conflicting observed labels

`HKU-CSD/CCDC_data.pkl.gz` contains 19,851 unique component MolBlocks. In the
current audit, parsed SMILES and MolBlocks agreed on formal charge and heavy-atom
count whenever both representations parsed successfully.

This means the original charge information is available. The charge-stripped
derived CSV/NPZ files must not be used as the source for corrected chemistry.
However, the historical assets discarded entry-level component multiplicity.
A licensed CCDC re-export is still required for reliable stoichiometry,
component provenance, and ambiguous or multivalent salts.

## Confirmed Legacy Defects

1. Charge symbols and SDF charge records were removed without repairing
   protonation, valence, or bond order.
2. A/B reversal happened before row-level splitting, allowing the same physical
   pair into both train and validation sets.
3. Multiple observations of the same unordered pair were not grouped before
   splitting.
4. Fine-tuning evaluated every epoch on the eventual external holdout.
5. The four-class fine-tuning notebook overwrote its best checkpoint with the
   final epoch.
6. Pretraining scheduling used training loss rather than validation loss.
7. The binary and four-class notebooks used different and partly stale dataset
   naming conventions.
8. Cross-molecule hydrogen-bond, pi-stacking, and contact candidates existed in
   featurization code but were disabled by the dataset builder defaults.
9. Padded zero nodes were converted to graph nodes by the legacy loader.
10. Failed conversions and filtering decisions were not represented by a
    durable rejection manifest.

## Freeze Rules

- Never edit a legacy artifact in place.
- Keep observed CSD structures separate from neutral-parent candidates and
  model-input structures.
- A component-level RDKit `ChargeParent` result is only a candidate. It is not
  an approved reactant parent until pair-level charge, proton, stoichiometry,
  valence, and uniqueness checks pass.
- Split by canonical unordered physical pair before A/B augmentation.
- Preserve every rejected or ambiguous row with a structured reason.
- Use the 50-pair external holdout only once for final locked evaluation.
- Report binary and four-class pretraining and fine-tuning as four separate
  model artifacts.
