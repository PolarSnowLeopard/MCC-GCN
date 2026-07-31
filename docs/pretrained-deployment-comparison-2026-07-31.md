# Pretrained Deployment Comparison (2026-07-31)

## Decision

Do not replace the production `MCC-GCN Pretrained v1` checkpoint with either
provisional corrected checkpoint. The corrected four-class candidate changes
the dominant predicted class but does not materially improve balanced
accuracy. The corrected binary candidate is also collapsed and is not
architecture-compatible with the current four-class production model.

The production pretrained checkpoint is not a valid fallback. Its BatchNorm
state shows only six tracked batches, and it collapses even on retained legacy
CCDC features. It was likely uploaded from an early or incomplete run rather
than from the 372-epoch notebook checkpoint. Keep the working fine-tuned model
available, but withdraw or clearly mark the pretrained model as unavailable
until a validated checkpoint is recovered or retrained.

## Feedback Dataset Identity

Two different datasets have been discussed:

- The original customer screenshots show a 12-row batch with API
  `CCOC(=O)c1ccnc(c1)C(=O)OCC`. This is not the locked external 64.
- The old frontend paper preset is exactly the locked external 64: two APIs,
  KPX and KPR, crossed with 32 coformers.

The effectively identical pretrained predictions were independently reproduced
on both the customer's 12-row batch and the locked frontend 64.

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

## Production Fine-Tuned v1

The production `MCC-GCN v1` model, model ID 2, was evaluated through the same
deployed RDKit inference path on the locked 50 pairs excluded from fine-tuning.
Production task ID: 339.

- Overall accuracy: 0.5800
- Balanced accuracy: 0.5628
- Prediction counts: negative 7, salt 10, cocrystal 11, solvate 22

Confusion matrix:

```text
[[ 5, 2, 2, 5],
 [ 0, 4, 3, 1],
 [ 0, 2, 4, 0],
 [ 2, 2, 2, 16]]
```

This agrees with the legacy evaluation's overall accuracy and confirms that the
deployed fine-tuned model is not exhibiting the pretrained model's total
collapse.

## Legacy CCDC In-Domain Check

The production pretrained checkpoint was also evaluated with the original
CCDC-derived features from `HKU_data_5_total_inbalance.npz`. The diagnostic set
contained 787 old validation rows:

- 198 negative rows from the historical validation split
- 589 positive-class orientation rows whose CSD identifiers did not occur in
  the historical training split

The checkpoint predicted solvate for all 787 rows:

- Overall accuracy: 0.1398
- Balanced accuracy: 0.2500
- Prediction counts: negative 0, salt 0, cocrystal 0, solvate 787

Removing padded nodes did not change any predicted class. Therefore the
production collapse cannot be explained only by external-distribution shift or
by replacing CCDC features with RDKit features at deployment.

Checkpoint-state evidence is consistent with the wrong pretrained artifact
having been published:

- `best_model.pth`: all five `num_batches_tracked` values are 6
- `best_FT_model.pth`: all five values are 358586

The notebook recorded the best pretrained validation BACC at epoch 372. About
358 thousand tracked batches is plausible for that training duration; six is
not. The notebook's reported 0.8607 validation BACC is itself optimistic
because the row-level split leaked A/B orientations, but that leakage does not
explain the production checkpoint's near-initial BatchNorm state.

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
