# Revision experiment status

Last updated: 2026-08-22. This is an internal execution record, not manuscript
or response-letter text.

## Ready results

| Reviewer request | Evidence now available | Current conclusion |
|---|---|---|
| Three external APIs | Paracetamol 42 pairs, theophylline 110 pairs, riluzole 19 pairs; target APIs excluded from pretraining; five-fold grouped evaluation; three seeds | Binary zero-shot/fine-tuned accuracy 0.886/0.898 and BAcc 0.737/0.824; four-class 0.533/0.647 and BAcc 0.388/0.734 |
| API similarity | Reproducible Morgan-fingerprint Tanimoto tables for legacy and three-API experiments | New target APIs are mutually dissimilar (0.0625-0.0952); nearest source API similarity averages 0.5087 |
| Split before order augmentation | Physical-pair grouped folds and locked A/B+B/A augmentation | No A/B copy crosses a train/test boundary |
| Input-order sensitivity | Orientation-specific probabilities and total-variation diagnostics | Historical KPXKPR-50 checkpoint has 1.0 class agreement and negligible probability discrepancy; order is not the current failure source |
| Actual fine-tuning parameter count | Counts for every task and unfreezing setting | Submitted four-class last-three-layer setting trains 25,412/134,340 parameters (18.92%); binary trains 6,498/23,906 (27.18%) |
| Distribution-aware chance baseline | Prevalence-weighted expected and majority-class accuracy for all evaluation sets | Four-class weighted chance is 0.3120 for KPXKPR-50, 0.2842 for KPXKPR-64, 0.4379 for Three-API-170 and 0.4199 for source validation |
| Minoxidil-only adaptation | Frozen four-class checkpoint, 20 Minoxidil pairs, all 64 KPX/KPR pairs, three seeds | Does not provide satisfactory transfer. Historical-padding accuracy/BAcc 0.245/0.184; corrected trimming 0.375/0.264; negative recall is 0 in all primary runs |
| KPXKPR-50 post-FT embedding | Graph, FC1, FC2, logits, probability and projected-space separation metrics | Submitted checkpoint does not produce clear class clusters. FC2 silhouette is 0.0062 and UMAP silhouette is -0.0032; the original positive claim cannot be retained unchanged |
| Dataset and target counts | Frozen design summary | Source pretraining has 22,307 physical pairs after excluding target APIs; target set has 170 pairs with full per-API/per-class counts |
| Pretraining undersampling protocol | Controlled full-data/effective-number versus task-specific undersampling comparison on the same target-excluded split | Code, fixed subset construction, feature build, three-seed runner and metric aggregation are ready; formal cluster results remain pending |

Detailed sources:

- `runs/revision-three-api/formal-multiseed-summary/target_api_summary.md`
- `runs/revision-three-api/tanimoto-v1/summary.json`
- `runs/revision-three-api/design-summary-v1/revision_design_summary.md`
- `docs/revision-minoxidil-only-results.md`
- `docs/revision-chance-baselines.md`
- `runs/revision-kpxkpr50-embedding/kpxkpr50_post_finetune_embedding.metrics.json`

## Implemented but missing formal result artifacts

| Experiment | Implementation | What remains |
|---|---|---|
| Fine-tuning learning curve | Nested, class-balanced 8/16/32/48-pair subsets for every fold and seed | A cluster run was reported complete, but its formal output directory has not been synchronized back; aggregate and verify before citation |
| Descriptor baselines | Order-invariant RDKit descriptors with SVM, RF and MLP; source zero-shot and target-supervised five-fold protocols | Run all models/tasks/seeds and synchronize the formal summary |
| Randomized-SMILES CNN | Independent two-branch CNN with physical-pair splitting before SMILES randomization and orientation averaging | Run all tasks/seeds and synchronize the formal summary; identify it as the implemented randomized-SMILES CNN, not the separately published DeepCocrystal software |

## Still required

| Reviewer request | Required action | Priority |
|---|---|---:|
| Negative-set and undersampling concern | Run the implemented controlled comparison and synchronize its aggregate table; report false-positive and class-specific behavior | High |
| Fine-tuning protocol disclosure | Correct the supplementary parameter table: executed class weights were `[1,1,1,2]`, not `[1,1,1,1]`; disclose that the historical code used the target holdout for per-epoch validation and replace this with leakage-free revision results | High |
| Learning-curve result | Recover/synchronize the completed cluster outputs and generate the final table/plot | High |
| Baseline results | Run and aggregate descriptor and CNN formal benchmarks | High |
| Dataset overlap figure | Render the already locked set relationships: KPXKPR-64 = 14 adaptation + 50 holdout; FT-34 = 20 Minoxidil + 14 KPX/KPR | Medium |
| Common solvent/hydrate analysis | Identify water/methanol/ethanol-related entries that are actually represented and report scope without inventing unperformed wet experiments | Medium |
| Attribution correction | Reassess the chemically equivalent phthalic-acid carboxyl groups and remove any claim that an unconditioned graph can identify which equivalent group physically interacts | Medium |
| Salt/cocrystal overlap explanation | Use confusion/embedding evidence and label-definition ambiguity; do not reuse the unsupported cluster-separation claim | Medium |
| Annotated CSD-derived dataset release | Obtain written CCDC redistribution approval before publishing CSD-derived SMILES/labels; public release remains blocked until then | External dependency |

## Decision points

1. The Minoxidil-only and embedding analyses are valid negative results. They
   should be used to narrow claims and acknowledge target-domain limitations,
   not hidden or presented as performance improvements.
2. The three-API experiment is a new experiment and may use the corrected
   leakage-free protocol. It must not be described as an exact rerun of the
   submitted KPX/KPR experiment.
3. Formal comparison tables must keep zero-shot, target-supervised five-fold,
   and fine-tuned MCC-GCN regimes separate. Training a baseline from scratch on
   target folds is not fine-tuning.
4. No test or target label may control learning-rate scheduling, checkpoint
   selection, hyperparameter choice, or subset choice in remaining runs.
