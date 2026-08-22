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

The official source is licensed under the Academic Software Licence. It must
remain a separate checkout and is not copied into or redistributed with this
repository.

```bash
git clone https://github.com/molML/deep-cocrystal.git \
  /workspace/deep-cocrystal

RUN_ID="revision-three-api-deepcocrystal-seed42"
python scripts/run_deepcocrystal_architecture_baseline.py \
  --data-root data/revision-three-api \
  --deepcocrystal-repo /workspace/deep-cocrystal \
  --output-dir "runs/$RUN_ID" \
  --seed 42 \
  --tensorboard-dir "/primus_oss/summary/mcc-gcn/$RUN_ID"
```
