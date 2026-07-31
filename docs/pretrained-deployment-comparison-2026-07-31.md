# Pretrained Deployment Comparison (2026-07-31)

## Decision

Replace the invalid production `MCC-GCN Pretrained v1` checkpoint with the
corrected four-class pretrained checkpoint. The replacement has strong
pair-disjoint in-domain validation results and removes the old checkpoint's
near-constant probability output. It remains weak on the external 64 and still
assigns one class to the customer's chemically similar 12-row batch. These
out-of-distribution limitations must not be presented as resolved by
deployment.

The production pretrained v1 checkpoint was not a valid fallback. Its
BatchNorm state showed only six tracked batches, and it collapsed even on
retained legacy CCDC features. It was likely uploaded from an early or
incomplete run rather than from the 372-epoch notebook checkpoint.

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
- Historical production model: built-in `MCC-GCN Pretrained v1`, model ID 1
- Historical production task ID: 338
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

Pair-disjoint in-domain validation:

- Overall accuracy: 0.8588
- Balanced accuracy: 0.7954
- Per-class recall: negative 0.9320, salt 0.9726, cocrystal 0.7718,
  hydrate/solvate 0.5052

Confusion matrix:

```text
[[ 192,    2,   12,   0],
 [   6, 1988,   30,  20],
 [  70,  166, 1590, 234],
 [   4,   32,   60,  98]]
```

Locked external 64:

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

The candidate does not repeat one exact class for every row. Its poor external
result and absence of negative predictions show a substantial distribution
shift between the corrected CCDC-derived pretraining data and the external
experimental pairs. This does not invalidate the pretrained checkpoint's
in-domain role, but it means the external use case still requires fine-tuning
or additional representative training data.

## Corrected Binary Pretrained Candidate

- Overall accuracy: 0.7344
- Balanced accuracy: 0.5000
- Confusion matrix: `[[0, 17], [0, 47]]`

It predicts every external pair as positive. It cannot replace the current
four-class model without changing the API contract and frontend labels.

## Deployment Compatibility

Legacy production applied padding by built-in model type:

- pretrained: 178 nodes
- fine-tuned: 70 nodes

Corrected packed-graph checkpoints were trained without padded nodes. The
application now stores an explicit inference profile per model instead of
inferring padding from model type:

- `MCC-GCN Pretrained v2`, model ID 1: large model, no padding, RDKit SMILES
  2D coordinates, covalent edges
- `MCC-GCN v1`, model ID 2: large model, legacy 70-node padding

## Production Deployment

Deployed on 2026-07-31 from `MCC-GCN-App` commit `051f1f6`.

- Checkpoint SHA256:
  `197c7a2533b0e01c38a93c3f3137c87f4d6b2f2c4ead2e0edcf356fa050acc26`
- Built-in model ID 1 was preserved and renamed to
  `MCC-GCN Pretrained v2`.
- The backend and Celery workers were restarted to clear cached model state.
- GitHub CI and CD completed successfully.

Post-deployment public API verification:

- Pretrained v2, locked external 64, task ID 340: 64/64 predicted classes
  matched the cluster evaluation; no row failed; accuracy 0.234375 and balanced
  accuracy 0.2515.
- Fine-tuned v1, locked holdout 50, task ID 341: no row failed; accuracy 0.5800
  and balanced accuracy 0.5628, exactly preserving the pre-deployment baseline.
- HTTPS homepage returned 200 and all Compose services remained running with no
  backend or Celery errors during the checks.

## Customer Feedback Reproduction After Deployment

The corrected 12-row customer batch was recovered from historical production
task ID 67. The legacy pretrained model predicted Solvate for all 12 rows.
Re-running the exact same inputs against Pretrained v2 produced task ID 342:

- all 12 rows completed without errors
- all 12 predicted labels were Cocrystal
- there were 10 distinct probability vectors because two chemical pairs were
  duplicated
- Cocrystal confidence ranged from 0.6640 to 0.9750

Therefore the old checkpoint's nearly constant four-class probabilities have
been fixed, but the customer's visible complaint that this particular batch
has one predicted label remains true. This is model behavior on a narrow
external series, not stale API output or reuse of one inference result.

The separate single-pair example from the customer screenshot was also
recovered from historical task IDs 22 and 34. It changed from legacy Solvate
probabilities `[0.2538, 0.2262, 0.2547, 0.2652]` to Salt with probability
0.9855 in post-deployment task ID 343. This confirms that Pretrained v2 does
not return one globally constant output for unrelated inputs.
