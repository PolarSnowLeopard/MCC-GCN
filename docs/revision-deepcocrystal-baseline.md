# DeepCocrystal binary baseline

The reviewer-requested DeepCocrystal comparison must not use the checkpoint
distributed by the official repository as an external-validation result. Its
training data already contain at least 24 Paracetamol, 10 Theophylline, and
five Riluzole pairs, and its published label is cocrystal-specific rather than
the manuscript's any-MCC-versus-negative binary label.

The minimal fair comparison therefore uses the unmodified official
DeepCocrystal architecture from commit
`2805197eff579844b6a45f71a67e2638ede7a14e`, retrained on the same frozen,
target-API-excluded source split as MCC-GCN. It retains all source pairs, uses
class weighting, applies randomized-SMILES and A/B-order augmentation only
after the physical-pair split, and averages the two input orders at inference.
Only seed 42 is run; no hyperparameter search or repeated-seed aggregate is
required.

The isolated environment uses TensorFlow 2.18.1 with the maintained legacy
`tf.keras` compatibility package because the official implementation was
written for TensorFlow 2.7.1. The official DeepCocrystal source itself remains
unmodified.

The official source is licensed under the Academic Software Licence. It must
remain a separate checkout and is not copied into or redistributed with this
repository.

```bash
cd /workspace/MCC-GCN
bash scripts/bootstrap_deepcocrystal_env.sh

RUN_ID="revision-three-api-deepcocrystal-seed42"
/workspace/venvs/deepcocrystal-tf218/bin/python \
  scripts/run_deepcocrystal_architecture_baseline.py \
  --data-root data/revision-three-api \
  --deepcocrystal-repo /workspace/deep-cocrystal \
  --output-dir "runs/$RUN_ID" \
  --seed 42 \
  --tensorboard-dir "/primus_oss/summary/mcc-gcn/$RUN_ID"
```
