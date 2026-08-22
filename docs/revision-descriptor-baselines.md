# RDKit descriptor baselines for the three-API experiment

The historical notebook contains an RBF SVM experiment based on 24 descriptors
from `CCG_Featurize.Cocrystal`. The saved descriptor matrices are unavailable,
and regenerating the same features requires the original crystal-structure
inputs and toolchain. The revision benchmark therefore keeps the historical
paper result unchanged and reports a separate, reproducible no-CCDC comparison.

## Representation

Sixteen RDKit 2D descriptors are computed for each molecular component. The
pair vector concatenates the component-wise mean and absolute difference,
yielding 32 order-invariant values. The complete descriptor list and library
versions are written to `experiment_config.json`.

## Protocol

- SVM, random forest, and MLP use the frozen physical-pair splits from the
  three-API experiment.
- The source zero-shot model is fitted only on the target-API-excluded
  pretraining split and evaluated on all 170 target pairs.
- The five-fold target-supervised model is trained from scratch on the 136
  physical pairs in each training fold and evaluated on its 34 held-out pairs.
- Class-balanced sample weights are computed from each training set.
- The target-supervised classical models are controls for limited target-domain
  supervision. They are not described as fine-tuned or pretrained models.
- Seed 42 is fixed for reproducibility; repeated-seed aggregation is not part
  of the minimal revision comparison.

## Formal command

```bash
RUN_ID="revision-three-api-descriptor-baselines-$(date +%Y%m%d-%H%M%S)"
python scripts/run_target_api_descriptor_baselines.py \
  --data-root data/revision-three-api \
  --output-dir "runs/$RUN_ID" \
  --tasks binary,four-class \
  --models svm,random_forest,mlp \
  --seeds 42 \
  --folds 5 \
  2>&1 | tee "$RUN_ID.log"
```

This experiment is CPU-oriented and can run while the GPU is occupied, subject
to cluster CPU and memory limits.
