# Pretrained Deployment Comparison (2026-07-31)

## Decision

Do not replace the production `MCC-GCN Pretrained v1` checkpoint with either
provisional corrected checkpoint. The corrected four-class candidate changes
the dominant predicted class but does not materially improve balanced
accuracy. The corrected binary candidate is also collapsed and is not
architecture-compatible with the current four-class production model.

## Evaluation Contract

- Dataset: the locked 64 external physical pairs
- True class counts: negative 17, salt 13, cocrystal 9, solvate 25
- Production path: authenticated batch API at `yufanwenshu.cn`
- Production model: built-in `MCC-GCN Pretrained v1`, model ID 1
- Production task ID: 338
- Candidate path: corrected packed-graph evaluation with matched A/B and B/A
  predictions averaged

## Production Pretrained v1

- Overall accuracy: 0.390625
- Balanced accuracy: 0.250000
- Prediction counts: negative 0, salt 0, cocrystal 0, solvate 64

Confusion matrix:

```text
[[ 0,  0,  0, 17],
 [ 0,  0,  0, 13],
 [ 0,  0,  0,  9],
 [ 0,  0,  0, 25]]
```

Production probability ranges across the 64 pairs were also narrow:

| Class | Minimum | Maximum |
| --- | ---: | ---: |
| Negative | 0.2519 | 0.2526 |
| Salt | 0.2248 | 0.2256 |
| Cocrystal | 0.2504 | 0.2507 |
| Solvate | 0.2711 | 0.2727 |

This reproduces the reported behavior that different inputs produce
effectively identical results.

## Corrected Four-Class Pretrained Candidate

Run:
`runs/provisional-full-seed42-20260731-020558/four-class/seed-42/pretrain`

- Overall accuracy: 0.234375
- Balanced accuracy: 0.2515
- Prediction counts: negative 0, salt 51, cocrystal 7, solvate 6

Confusion matrix:

```text
[[ 0, 14, 3, 0],
 [ 0, 11, 0, 2],
 [ 0,  9, 0, 0],
 [ 0, 17, 4, 4]]
```

The candidate does not repeat one exact class for every row, but it remains
collapsed, never predicts the negative class, and has lower overall accuracy.
The balanced-accuracy improvement of 0.0015 is not meaningful.

## Corrected Binary Pretrained Candidate

- Overall accuracy: 0.7344
- Balanced accuracy: 0.5000
- Confusion matrix: `[[0, 17], [0, 47]]`

It predicts every external pair as positive. It cannot replace the current
four-class model without changing the API contract and frontend labels.

## Deployment Compatibility

Production currently applies legacy padding by built-in model type:

- pretrained: 178 nodes
- fine-tuned: 70 nodes

Corrected packed-graph checkpoints were trained without padded nodes. A future
deployment must store an explicit inference profile per model; simply replacing
the production checkpoint file would incorrectly apply 178-node legacy
padding to the corrected model.
