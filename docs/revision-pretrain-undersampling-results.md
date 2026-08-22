# Pretraining undersampling ablation results

The formal target-API pretraining ablation completed all 12 planned runs: two
tasks, two training strategies, and model seeds 42/43/44. All runs used the
same target-excluded source split, source validation set, 170-pair three-API
target set, model architecture, and optimization settings.

## Main results

Values are mean +/- sample standard deviation over three model seeds.

| Task | Strategy | Training pairs | Source validation BAcc | Target accuracy | Target BAcc | Target macro F1 | Negative FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Binary | Full data + effective-number weighting | 20,076 | 0.947 +/- 0.006 | 0.871 +/- 0.026 | 0.768 +/- 0.037 | 0.763 +/- 0.041 | 0.383 +/- 0.057 |
| Binary | Random undersampling + uniform loss | 1,806 | 0.939 +/- 0.005 | 0.855 +/- 0.038 | 0.849 +/- 0.018 | 0.780 +/- 0.040 | 0.160 +/- 0.021 |
| Four-class | Full data + effective-number weighting | 20,076 | 0.819 +/- 0.005 | 0.569 +/- 0.019 | 0.423 +/- 0.013 | 0.401 +/- 0.008 | 0.481 +/- 0.098 |
| Four-class | Random undersampling + uniform loss | 3,464 | 0.782 +/- 0.003 | 0.402 +/- 0.077 | 0.474 +/- 0.010 | 0.371 +/- 0.028 | 0.333 +/- 0.037 |

The target support is 27 negative and 143 positive pairs for the binary task.
For four-class evaluation it is 27 negative, 17 salt, 106 cocrystal, and 20
hydrate/solvate pairs.

## Interpretation

For binary pretraining, undersampling discarded 91.0% of the source training
pairs but improved target balanced accuracy by 0.081 and reduced the negative
false-positive rate by 0.222. It slightly reduced overall accuracy by 0.016 and
source validation balanced accuracy by 0.008. The main trade-off was increased
negative recall (0.840 versus 0.617) at the cost of positive recall (0.858
versus 0.918).

For four-class pretraining, undersampling discarded 82.7% of the source
training pairs. Retaining all observations improved source validation balanced
accuracy by 0.037, target accuracy by 0.167, target macro F1 by 0.029, and
cocrystal recall from 0.302 to 0.689. Undersampling improved target balanced
accuracy by 0.051, negative recall from 0.519 to 0.667, and hydrate/solvate
recall from 0.483 to 0.867, but it overpredicted hydrate/solvate and sharply
reduced cocrystal recall. Salt recall remained near zero under both zero-shot
strategies, so this ablation does not resolve the source-to-target salt shift.

The per-API results show the same trade-off rather than one strategy winning
everywhere. In four-class evaluation, full-data training was materially better
for theophylline and riluzole, whereas undersampling improved paracetamol
balanced accuracy. The binary riluzole result was perfect under both strategies.

These results support retaining all accepted source observations for the
four-class revision model instead of discarding the majority of the dataset.
For binary classification, random undersampling remains competitive when
negative-class sensitivity is the primary objective. The manuscript response
should report both accuracy and class-balanced metrics and describe this as a
task-dependent trade-off. Three seeds quantify run-to-run variation but are not
enough to claim a formal significance test.

## Reproducibility record

- Run directory: `runs/revision-three-api-pretrain-ablation-formal`
- Summary JSON SHA-256:
  `b2eb179667fa199cd0adc1a32fc7ab61ad9db01e43256ff674e4f53fdb121c12`
- Per-run CSV SHA-256:
  `d83e14037c5d464d904eecd576b23d6da5ceb215ca9b8ace7c2db19a9cd62951`
- Fixed subset seed: 42
- Model seeds: 42, 43, 44
- Planned runs present: 12/12
- Missing result files: 0
