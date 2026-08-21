# Distribution-aware chance baselines

Reviewer 2 correctly noted that 25% is not the appropriate random reference
for an imbalanced four-class test set. The distribution-weighted expected
accuracy is

`sum(p_c^2)`,

where `p_c` is the observed prevalence of class `c`. This is the expected
accuracy of a random predictor that samples labels from the same class
distribution. Majority-class accuracy is reported separately as a stronger
constant-prediction reference.

| Dataset | Task | N | Weighted expected accuracy | Majority-class accuracy |
|---|---|---:|---:|---:|
| KPXKPR-50 | Binary | 50 | 0.5968 | 0.7200 |
| KPXKPR-50 | Four-class | 50 | 0.3120 | 0.4400 |
| KPXKPR-64 | Binary | 64 | 0.6099 | 0.7344 |
| KPXKPR-64 | Four-class | 64 | 0.2842 | 0.3906 |
| Three-API-170 | Binary | 170 | 0.7328 | 0.8412 |
| Three-API-170 | Four-class | 170 | 0.4379 | 0.6235 |
| Source validation | Binary | 2,231 | 0.9144 | 0.9552 |
| Source validation | Four-class | 2,231 | 0.4199 | 0.4576 |

The submitted fine-tuned four-class KPXKPR-50 accuracy of 0.58 remains above
both the 0.312 weighted random expectation and the 0.44 majority baseline.
The revised three-API four-class zero-shot and fine-tuned accuracies of 0.533
and 0.647 are above the 0.438 weighted expectation, but only the fine-tuned
result exceeds the 0.624 majority-class baseline. Balanced accuracy and
per-class metrics therefore remain necessary and should lead the comparison.

The binary source and three-API datasets are strongly positive-dominated, so
raw accuracy alone is particularly misleading. For example, the three-API
binary majority baseline is already 0.841. The reported binary accuracy should
always be accompanied by balanced accuracy, class recall, and F1.

Reproduce the table with:

```bash
python scripts/summarize_revision_chance_baselines.py \
  --output-dir runs/revision-three-api/chance-baselines-v1
```

The command writes CSV, JSON, and Markdown outputs with the exact class counts.
