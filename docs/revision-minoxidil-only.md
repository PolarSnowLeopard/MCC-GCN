# Minoxidil-only KPX/KPR-64 experiment

Reviewer 2 asks whether structural analogs alone can support prediction for an
unseen API. This experiment fine-tunes the submitted four-class MCC-GCN only
on the 20 locked Minoxidil physical pairs and evaluates all 64 KPX/KPR pairs.
No KPX or KPR label is used during fine-tuning.

## Frozen protocol

- Initial model: submitted four-class checkpoint `checkpoints/best_model.pth`.
- Fine-tuning set: 20 Minoxidil physical pairs, five per class.
- Evaluation set: all 64 KPX/KPR physical pairs (17 negative, 13 salt, nine
  cocrystal, and 25 hydrate/solvate).
- Augmentation: split identity is locked at the physical-pair level before A/B
  and B/A orientations are loaded.
- Inference: mean of the A/B and B/A class probabilities.
- Fine-tuning: 50 epochs, batch size 16, Adam learning rate `3e-4`, weight
  decay `0.3`, executed class weights `[1,1,1,2]`, final three dense layers
  trainable.
- Primary submitted seed: 12. Seeds 42 and 43 quantify stochastic variation.

The executed notebook used the KPX/KPR holdout loss for scheduling and its
balanced accuracy for checkpoint selection, then overwrote the selected file
with the epoch-50 checkpoint. Reusing target labels would invalidate the
Minoxidil-only test. This runner therefore keeps the submitted 50-epoch final
checkpoint convention and monitors only Minoxidil training loss for learning
rate scheduling. Target labels are never passed to optimization, scheduling,
checkpoint selection, or any other training decision.

The default `legacy-padded` mode faithfully uses the submitted dense feature
artifacts. It preserves the historical width of 80 nodes for fine-tuning and
the former 14 target-adaptation pairs, and 70 nodes for the original 50-pair
holdout. The optional `trimmed` mode is a separately labeled sensitivity
analysis of the corrected loader; it must not be mixed with the submitted
compatibility result.

## Run

```bash
cd /workspace/MCC-GCN
OUT=runs/revision-minoxidil-only-legacy-$(date +%Y%m%d-%H%M%S)
python scripts/run_minoxidil_only_experiment.py \
  --input-mode legacy-padded \
  --seeds 12,42,43 \
  --tensorboard-dir /primus_oss/summary/mcc-gcn/revision-minoxidil-only \
  --output-dir "$OUT"
```

For the corrected-loader sensitivity analysis:

```bash
OUT=runs/revision-minoxidil-only-trimmed-$(date +%Y%m%d-%H%M%S)
python scripts/run_minoxidil_only_experiment.py \
  --input-mode trimmed \
  --seeds 12,42,43 \
  --tensorboard-dir /primus_oss/summary/mcc-gcn/revision-minoxidil-only \
  --output-dir "$OUT"
```

Each run records hashes of every source artifact, per-seed checkpoints,
training histories, orientation-resolved probabilities, confusion matrices,
and aggregate mean/sample-standard-deviation metrics in `summary.json`.
