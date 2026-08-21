# API Tanimoto analysis

This analysis addresses the reviewer request to assess the structural
similarity of the APIs used for fine-tuning and provides the same context for
the three-API revision experiment.

## Method

- Fingerprint: Morgan/ECFP4, radius 2, 2,048 bits, chirality enabled
- Similarity: Tanimoto coefficient
- Pair similarity: the largest mean component similarity across the two
  possible assignments between a target pair and a pretraining pair
- Target benchmark: 170 physical pairs
- Pretraining reference: 22,307 physical pairs after excluding all records
  containing Paracetamol, Theophylline, or Riluzole

The API-to-API calculation uses one fixed reference structure for each API.
The target-to-pretraining calculation uses the actual standardized structures
present in every benchmark row, including observed ionic or tautomeric forms.

## Fine-tuning APIs

| API 1 | API 2 | Tanimoto |
|---|---|---:|
| Minoxidil | Kopyxil | 0.111 |
| Minoxidil | Kopyrrol | 0.415 |
| Kopyxil | Kopyrrol | 0.333 |

The original fine-tuning APIs therefore span low to moderate structural
similarity rather than forming a set of near-identical analogues.

## Revision APIs

| API 1 | API 2 | Tanimoto |
|---|---|---:|
| Paracetamol | Theophylline | 0.095 |
| Paracetamol | Riluzole | 0.063 |
| Theophylline | Riluzole | 0.075 |

The three revision APIs are mutually dissimilar by this fingerprint measure.

## Target-to-pretraining similarity

| Measure across 170 target pairs | Mean | Median | Exact-match fraction |
|---|---:|---:|---:|
| API component to nearest pretraining component | 0.509 | 0.457 | 0.000 |
| Coformer to nearest pretraining component | 0.902 | 1.000 | 0.771 |
| Complete pair to nearest pretraining pair | 0.599 | 0.609 | 0.000 |

Complete-pair nearest-neighbour similarity by class was 0.668 for negative,
0.380 for salt, 0.616 for cocrystal, and 0.601 for hydrate or solvate pairs.
Salt is the most structurally shifted class in this benchmark.

## Reproduction

```bash
python scripts/analyze_api_tanimoto.py \
  --target-pairs data/revision-three-api/four_class_target_physical_pairs.csv \
  --pretraining-pairs data/revision-three-api/four_class_pretrain_excluding_target_apis.csv \
  --output-dir runs/revision-three-api/tanimoto-v1
```

The command writes the full API similarity matrix, long-form pairwise values,
row-level nearest neighbours, hashes, software version, and aggregate summary.
