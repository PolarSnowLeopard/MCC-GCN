# Data Quality Rebuild

The legacy CSV and NPZ artifacts remain useful for reproducing the published
baseline, but they must not be treated as the source of truth for new training.

The frozen baseline and first reproducible audit are recorded in
`docs/legacy-baseline-freeze.md` and `docs/legacy-data-audit.md`.

## Confirmed Legacy Risks

1. CCDC component SMILES were "neutralized" with string replacement:
   `smiles.replace("+", "").replace("-", "")`. This can create invalid valence
   states and radicals instead of chemically valid neutral parents.
2. SDF charge records were removed by editing V2000 text lines. Atom valence,
   hydrogen count, and bond order were not repaired.
3. A/B and B/A rows were generated before random train/validation splitting.
   The same unordered molecular pair can therefore occur in both splits.
4. The same unordered pair can have multiple CSD outcomes. Some legacy rows
   also occur in both the CCG-Net negative set and the CSD positive set.
5. Salt/cocrystal labels were assigned from component formal charge alone.
   Solvate/hydrate labels were assigned from component count and water
   presence, while solvent and experimental conditions were omitted from the
   model input.
6. Failed conversions and filtering exceptions were often skipped without a
   durable rejection table.

## Immediate Audit

The audit command does not require CCDC:

```bash
python scripts/audit_data.py \
  data/csd_pairs.csv data/negative_pairs.csv \
  --output-dir runs/data-audit
```

It emits:

- `audit_summary.json`: counts of invalid structures, duplicates, and conflicts.
- `audited_rows.csv`: every source row plus canonical structure metadata.
- `quarantined_rows.csv`: invalid, radical, disconnected, unsupported, or
  conflicting rows.
- `conflicting_pairs.csv`: one row per unordered pair with incompatible labels.
- `clean_pair_candidates.csv`: one canonical row per unambiguous unordered pair.
- `source_pair_overlap.csv`: pair-level overlap between every input source.

The last file is only a mechanically clean candidate. It is not automatically
approved training data. Chemistry and outcome labels still require review.

Assess component-level neutralization candidates and pair-level charge/hydrogen
conservation separately:

```bash
python scripts/audit_neutralization_candidates.py \
  data/source/HKU_data_4_reactions.csv \
  --component-molblocks data/source/CCDC_data.pkl.gz \
  --output-dir runs/neutralization-audit
```

`balanced_monovalent_candidate` means only that:

- RDKit produced two neutral components.
- Each component's hydrogen change is the inverse of its formal charge.
- Atom-local reverse proton-transfer enumeration produced a unique structure.
- The local structure agrees with RDKit `ChargeParent`.

`balanced_multivalent_candidate` passes the same mechanical checks but requires
stronger stoichiometry and site review. Neither status approves a reactant
structure. Candidate pairs with conflicting labels are emitted separately and
cannot enter a training split.

## Rebuild Boundary

CCDC-dependent work must run once on a licensed workstation. Everything after
the raw export must run without CCDC on macOS and the company GPU cluster.

### Licensed CCDC workstation

Export an immutable record for every CSD entry:

- CSD version and extraction timestamp
- refcode/identifier and refcode family
- original raw label and label provenance
- all component SMILES, InChI, InChIKey, formal charges, stoichiometry, and
  original 3D mol blocks
- 3D/disorder flags and every filter result
- structured rejection code and error details

Do not neutralize structures during extraction. Do not discard solvent
components. Store the export in private OSS because CSD structure redistribution
may be license-restricted.

The exporter can be sharded so an interrupted run only needs to repeat one
shard:

```bash
python scripts/ccdc/export_manifest.py \
  --manifest data/source/csd_identifier_manifest.csv \
  --output data/raw/csd_entries.part-00.jsonl.gz \
  --shard-index 0 \
  --num-shards 8
```

Run shard indices `0` through `7`. Each shard writes a checksum metadata file
and a separate rejection file. A non-empty rejection file causes a non-zero
exit status so failed entries cannot be overlooked.

### CCDC-free processing

Build separate representations instead of overwriting the observed structure:

- `observed_structure`: exact CSD charge and protonation state
- `neutral_parent_candidate`: deterministic RDKit candidate, never automatic
  approval for a charged component
- `model_structure`: explicitly selected representation used for features
- `standardization_warnings`: every transformation and failure

Run component standardization on the Mac or GPU cluster:

```bash
python scripts/standardize_csd_export.py \
  data/raw/csd_entries.part-*.jsonl.gz \
  --output-dir data/interim/csd-standardized
```

The command uses CCDC MolBlocks first and only falls back to InChI or SMILES
when necessary. It writes separate observed structures and neutralization
candidates, together with a candidate review status. A charged component is
never promoted to a model structure by this command. It fails with a non-zero
status if any component cannot be standardized.

Group and split data by canonical unordered parent pair before any A/B
augmentation. Preserve all CSD observations as evidence; do not silently choose
one polymorph or outcome. Conflicting outcomes must be quarantined or modeled as
condition-dependent evidence.

## Label Policy Required Before Training

- A CCG-Net failed cocrystallization is evidence for `cocrystal_negative`, not
  evidence that salt, hydrate, or solvate formation is impossible.
- Salt/cocrystal assignment needs an explicit, documented protocol. Formal
  charge alone is insufficient; ambiguous cases should be retained as such.
- A solvate outcome depends on solvent and preparation conditions. Either add
  these inputs, restrict the class to hydrate with a stated scope, or report the
  pair-only task as an association rather than deterministic outcome prediction.
- Binary and four-class datasets should be generated from one evidence table
  with separate task-specific label functions and manifests.

## Versioned Outputs

Each rebuild should produce:

```text
data/raw/csd_entries.jsonl.gz
data/interim/components.parquet
data/interim/pair_evidence.parquet
data/curated/binary_pairs.parquet
data/curated/four_class_pairs.parquet
data/splits/<task>/<split_manifest>.csv
data/manifests/<dataset_version>.json
data/rejects/<dataset_version>.parquet
```

The manifest must include source checksums, CSD/RDKit versions, curation rules,
random seeds, class counts, pair-overlap checks, and output checksums.
