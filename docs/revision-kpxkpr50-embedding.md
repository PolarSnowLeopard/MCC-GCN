# KPXKPR-50 post-fine-tuning embedding analysis

Reviewer 2 requested that the original Figure 4c be removed and replaced with
a clustering analysis of the 50 protected KPX/KPR test pairs after
fine-tuning. The original panel projected the CSD background and displayed the
50 target samples only in small enlarged windows, which did not directly show
their class separation.

The diagnostic workflow extracts graph, dense-layer, logit, and probability
representations from the submitted four-class fine-tuned checkpoint. It
averages matched A/B and B/A representations and projects the 50 physical pairs
into two dimensions. Point color denotes the experimental class; an `X` marker
identifies a misclassification. The workflow reports silhouette,
Calinski-Harabasz, and Davies-Bouldin statistics for every model space and the
two-dimensional projection, so the plotted layer cannot be selected solely for
an attractive visual result.

The historical checkpoint must be evaluated with its original 70-node padded
input convention. Using the corrected padding-free loader changes the model's
effective graph readout and does not reproduce the submitted result.

```bash
python scripts/plot_kpxkpr50_embedding.py \
  --model checkpoints/best_FT_model.pth \
  --test-data-1 data/HKU_data_6_experiment_1.npz \
  --test-data-2 data/HKU_data_6_experiment_2.npz \
  --input-mode legacy-padded \
  --method umap \
  --representation fc2 \
  --seed 42 \
  --output-dir runs/revision-kpxkpr50-embedding
```

The classification output should reproduce the submitted KPXKPR-50 result
(accuracy 0.58) before the replacement panel is used in the revision.

The reproduced historical checkpoint does not currently support the submitted
claim of clearly separated target-domain clusters: its standardized `fc2`
silhouette is approximately 0.006, and the other internal representations are
similarly weak. The generated panel is therefore a diagnostic artifact, not a
positive replacement figure. A revised claim requires either a newly trained
model that improves separation under the same protected test protocol or a
change from cluster-separation language to an honest decision-boundary/error
analysis.
