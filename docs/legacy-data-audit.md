# Legacy Data Audit

This is the first reproducible audit snapshot taken before corrected dataset
construction. The machine-readable results are in
`data/manifests/legacy-data-audit-v1.json`.

## Main Finding

The original CSD chemistry is substantially recoverable, but the derived
training tables are not suitable as corrected-data inputs.

The historical `HKU-CSD/HKU_data_4_reactions.csv` contains 34,162 CSD rows and
still preserves charged component SMILES. Of those rows, 21,494 have a nonzero
component charge. The associated `CCDC_data.pkl.gz` contains 19,851 unique
component MolBlocks.

By contrast, the derived pretraining table has no remaining formal charges:

| Dataset | Rows | Invalid structures | Radical rows | Charged rows |
| --- | ---: | ---: | ---: | ---: |
| Original CSD pairs | 34,162 | 930 | 528 | 21,494 |
| Derived reaction table | 34,618 | 20,781 | 19,415 | 0 |
| Derived pretraining table | 69,236 | 41,562 | 38,830 | 0 |

This is consistent with charge symbols and SDF charge records being removed
without chemically repairing valence or protonation.

## Pair-Aware Recovery Audit

The original table contains 21,529 salt observations. The two most frequent
ordered charge patterns are `-1/+1` (8,140 rows) and `+1/-1` (6,999 rows).
Together they cover 15,139 rows, or 70.32% of the salt observations, but charge
balance alone is not sufficient to approve a neutral parent.

The CCDC-free pair audit uses the original component MolBlocks where possible,
falls back to the preserved SMILES, checks charge and hydrogen changes for each
component, and independently enumerates atom-local reverse proton transfers.
It excludes charge-separated resonance sites, such as a nitro oxygen, when
other sites are sufficient to explain the component's net charge. Its current
result is:

- 10,377 unique, locally consistent monovalent salt candidates
- 625 unique, locally consistent multivalent salt candidates
- 189 salt rows with multiple non-equivalent local neutralization candidates
- 81 salt rows with no valid atom-local candidate
- 36 salt rows where `ChargeParent` and the local candidate disagree
- 12,435 originally net-neutral pair observations retained without
  neutralization
- 3,567 rows requiring component stoichiometry
- 4,765 rows containing an unresolved charged component
- 2,023 rows with at least one unparseable component
- 61 rows whose proposed hydrogen changes do not match formal charges

In total, 23,437 rows form 21,978 mechanically eligible unordered candidate
pairs. Candidate canonicalization reveals 241 unordered pairs with conflicting
labels, affecting 571 rows. These conflicts must be resolved or quarantined
before training.

The mechanically balanced salt candidates are still not approved reactants.
Site ambiguity, tautomer choice, permanent ions, and multicomponent
stoichiometry remain chemistry review gates. The historical pair CSV and
component dictionary do not retain reliable entry-level multiplicity, so the
unbalanced cases require a new licensed CCDC export or explicit exclusion.

## Split Leakage Replay

The historical four-class notebook used:

```python
train_test_split(
    range(len(dataset)),
    test_size=0.1,
    random_state=42,
)
```

Replaying that split over the 69,236-row derived pretraining table gives:

- 62,312 training rows
- 6,924 validation rows
- 6,276 validation rows whose unordered molecular pair is also in training
- 90.64% validation-pair overlap
- only 323 unordered pairs appearing exclusively in validation

The overlap follows directly from adding A/B and B/A rows before splitting.
Corrected splits must group the canonical unordered physical pair before any
order augmentation.

## Overlap And Conflicts

- The original CSD table contains 194 valid unordered pairs with conflicting
  observed labels.
- The 1,041 unique negative pairs overlap 13 CSD pairs.
- The initial `4095108` fine-tuning CSV contains 54 physical pairs and overlaps
  34 of the 64 external pairs. It is not the final corrected fine-tuning set.
- The final protocol uses 34 physical fine-tuning pairs: 20 minoxidil pairs and
  14 external pairs. The remaining 50 external pairs must be the locked final
  holdout.

Conflicting observations are evidence, not automatically bad rows. They must be
retained in an evidence table and resolved by a documented task-specific label
policy rather than silently deduplicated.

## Reproduction

Create a CCDC-free RDKit environment and run:

```bash
python scripts/audit_data.py \
  /path/to/HKU-CSD/HKU_data_4_reactions.csv \
  /path/to/HKU_data_3_real_neg_data.csv \
  --output-dir runs/data-audit/original-v1
```

The exact input hashes and software versions used for this snapshot are stored
in `data/manifests/legacy-data-audit-v1.json`.

Reproduce the pair-aware neutralization audit with:

```bash
python scripts/audit_neutralization_candidates.py \
  /path/to/HKU-CSD/HKU_data_4_reactions.csv \
  --component-molblocks /path/to/HKU-CSD/CCDC_data.pkl.gz \
  --output-dir runs/data-audit/neutralization-v1
```
