# Minoxidil-only KPX/KPR-64 results

## Question

Reviewer 2 asked whether satisfactory KPX/KPR-64 performance can be obtained
when fine-tuning contains only the structural analog Minoxidil. The submitted
four-class pretraining checkpoint was fine-tuned on the 20 locked Minoxidil
pairs (five per class), with no KPX or KPR pair used for training, scheduling,
checkpoint selection, or other adaptation.

## Protocol audit

The executed historical notebook and the current supplementary draft disagree
on one material parameter. The notebook used class weights `[1,1,1,2]`; the
draft states `[1,1,1,1]`. Results below use the executed notebook weights.

The notebook also evaluated the 50-pair target holdout every epoch, used its
loss for learning-rate scheduling and its balanced accuracy for checkpoint
selection, and then overwrote the selected checkpoint with the epoch-50 model.
That target-label dependency cannot be used for this reviewer experiment. The
revised run therefore retains the fixed 50 epochs and epoch-50 checkpoint, but
monitors only Minoxidil training loss for scheduling. Target labels are used
only for fixed-protocol evaluation and never affect training decisions.

Other frozen settings are batch size 16, Adam learning rate `3e-4`, weight
decay `0.3`, final three dense layers trainable, and seeds 12, 42, and 43.
A/B and B/A probabilities are averaged at inference.

## Results

| Input handling | Stage | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Historical padding | Pretrained | 0.3906 | 0.2500 | 0.1404 |
| Historical padding | Minoxidil-only FT | 0.2448 +/- 0.0180 | 0.1837 +/- 0.0115 | 0.1529 +/- 0.0077 |
| Corrected trimming | Pretrained | 0.3906 | 0.2500 | 0.1404 |
| Corrected trimming | Minoxidil-only FT | 0.3750 +/- 0.0271 | 0.2644 +/- 0.0436 | 0.2034 +/- 0.0551 |

Mean per-class recall across the three fine-tuning seeds:

| Input handling | Negative | Salt | Cocrystal | Hydrate/solvate |
|---|---:|---:|---:|---:|
| Historical padding | 0.0000 | 0.0769 | 0.1111 | 0.5467 |
| Corrected trimming | 0.0000 | 0.1538 | 0.0370 | 0.8667 |

The pretrained model predicts all 64 pairs as hydrate/solvate. Fine-tuning on
Minoxidil alone does not recover the negative class in any seed. Correctly
trimming padded nodes improves the aggregate result relative to historical
padding, but it does not resolve the target-domain failure. Prediction
agreement between A/B and B/A is 1.0 in every run, so input order is not the
cause of this result.

For context, the submitted 34-pair adaptation used 20 Minoxidil pairs plus 14
KPX/KPR pairs and reported 0.58 accuracy on the remaining 50 pairs. That result
is not directly comparable to the 64-pair Minoxidil-only result because its
training and test sets differ. The contrast nevertheless shows that the 14
target-domain pairs, rather than structural analogy alone, were essential to
the reported domain adaptation.

## Interpretation for revision

The available evidence does not support a claim that Minoxidil-only
fine-tuning provides satisfactory four-class performance for unseen KPX/KPR
pairs. The defensible response is to acknowledge this limitation: pretraining
plus a structural analog does not remove the need for a small amount of
target-specific labeled data under the current model and dataset. The result
also rules out padding and input order as sufficient explanations for the
failure.

The historical binary pretraining checkpoint was not preserved in the paper
repository, deployment repository, VPN archive, or OSS baseline archive.
Therefore, an exact same-version binary Minoxidil-only fine-tuning result
cannot be reconstructed from the frozen artifacts.

## Reproducibility

Primary local output directories (not committed because they contain model
checkpoints):

- `runs/revision-minoxidil-only-legacy-weighted-20260822`
- `runs/revision-minoxidil-only-trimmed-weighted-20260822`

Each directory contains the source-artifact hashes, configuration, per-seed
training history, checkpoints, orientation-resolved probabilities, confusion
matrices, and `summary.json` aggregate statistics. The reusable command and
protocol are documented in `docs/revision-minoxidil-only.md`.
