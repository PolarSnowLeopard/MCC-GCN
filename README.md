# MCC-GCN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19612214.svg)](https://doi.org/10.5281/zenodo.19612214)

**Multi-Component Crystal Graph Convolutional Network** — An interpretable graph learning framework for multicomponent crystal classification and discovery.

MCC-GCN extends cocrystal prediction from binary classification to **four-class crystal form prediction** (cocrystal, salt, solvate, negative), and provides **gradient-based interpretability** at atomic resolution.

## Highlights

- **Four-class classification**: Simultaneously distinguishes cocrystals, salts, solvates, and negative outcomes
- **Transfer learning**: Pretrained on 34,000+ CSD entries, fine-tuned with only 34 samples for novel APIs
- **No CCDC required**: Pre-computed features and trained weights are provided — prediction works with SMILES input only
- **Interpretable**: Gradient-based attribution identifies key functional groups driving crystal formation

## System Requirements

- **Operating system:** Linux (tested on Ubuntu 22.04), macOS, or Windows with WSL
- **Python:** 3.9+
- **Hardware:** No non-standard hardware required. CPU is sufficient for inference and fine-tuning. GPU (NVIDIA CUDA) is recommended for pre-training.
- **Dependencies:** PyTorch, PyTorch Geometric, RDKit, OpenBabel, scikit-learn, NumPy, SciPy, pandas (see `requirements.txt` for full list with versions)
- **Typical install time:** ~10 minutes on a normal desktop computer (including conda environment and dependency installation)

## Installation

```bash
conda create -n mcc-gcn python=3.9 -y
conda activate mcc-gcn

# PyTorch (adjust cuda version as needed, or use cpuonly)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
# or for CUDA: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# PyTorch Geometric
pip install torch-geometric

# RDKit & OpenBabel
conda install -c conda-forge rdkit openbabel -y

# Install MCC-GCN
git clone https://github.com/PolarSnowLeopard/MCC-GCN.git
cd MCC-GCN
pip install -e .
```

## Data and Model Weights

### Included in this repository

| File | Size | Description |
|------|------|-------------|
| `checkpoints/best_model.pth` | 543 KB | Pretrained model weights (CSD dataset) |
| `checkpoints/best_FT_model.pth` | 544 KB | Fine-tuned model weights (paper results) |
| `data/HKU_data_6_FT_minoxidil_balanced_with_exp.npz` | 7.4 MB | Fine-tuning dataset (34 samples) |
| `data/HKU_data_6_experiment_1.npz` | 4.3 MB | KPXKPR-50 test set (A-B order) |
| `data/HKU_data_6_experiment_2.npz` | 4.3 MB | KPXKPR-50 test set (B-A order) |

### External download (optional, for pre-training only)

| File | Size | Description |
|------|------|-------------|
| `HKU_data_5_total_inbalance.npz` | ~34 GB | Pre-training features (34,621 samples from CSD) |

> **Download link:** Available via Zenodo upon publication. The code release is archived at [https://doi.org/10.5281/zenodo.19612214](https://doi.org/10.5281/zenodo.19612214).
>
> Place the file in `data/` after downloading.

### CCDC-dependent files (optional)

The following files are only needed if you want to re-extract molecular features from raw crystal structures. This requires a [CCDC CSD Python API](https://www.ccdc.cam.ac.uk/solutions/software/csd-python-api/) license.

| File | Description |
|------|-------------|
| `HKU_data.pkl.gz` | Merged mol blocks for feature extraction |
| `CCDC_data.pkl.gz` | CSD mol blocks |

## Quick Start

### Single Prediction (SMILES input, no CCDC needed)

```bash
python scripts/predict.py \
    --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "OC(=O)CC(=O)O" \
    --model checkpoints/best_FT_model.pth
```

Expected output (~5 seconds on a normal desktop CPU):

```
Building molecular graph...

Prediction: solvate
Probabilities:
  [0] negative:     0.0010
  [1] salt:         0.0001
  [2] cocrystal:    0.0007
  [3] solvate:      0.9982
```

Other input modes:

```bash
# From CAS numbers (requires internet for PubChem lookup)
python scripts/predict.py \
    --cas "58-08-2" "141-82-2" \
    --model checkpoints/best_FT_model.pth

# From SDF files
python scripts/predict.py \
    --sdf mol1.sdf mol2.sdf \
    --model checkpoints/best_FT_model.pth
```

### Evaluate an Existing Checkpoint

```bash
python scripts/evaluate.py \
    --model checkpoints/best_FT_model.pth \
    --test-data-1 data/HKU_data_6_experiment_1 \
    --test-data-2 data/HKU_data_6_experiment_2
```

> **Important:** This branch trims padded nodes before converting NPZ features
> to PyG graphs. The bundled checkpoint was trained with the legacy loader, so
> its historical paper result is a legacy-padding baseline rather than the
> expected result for this fixed-loader branch. Use the retraining workflow below
> to produce a new checkpoint.

Historical legacy-padding baseline (~30 seconds on a normal desktop CPU):

```
Overall Accuracy: 0.5800

Confusion Matrix:
[[ 6  2  2  4]
 [ 0  4  3  1]
 [ 0  1  5  0]
 [ 2  2  4 14]]
Class 0 Accuracy: 0.4286
Class 1 Accuracy: 0.5000
Class 2 Accuracy: 0.8333
Class 3 Accuracy: 0.6364

Results saved to prediction_results.csv
```

These historical results correspond to the fine-tuned MCC-GCN row in **Table 2**
and the confusion matrix in **Figure 3d** of the paper.

### Fine-tuning

```bash
python scripts/finetune.py \
    --data data/HKU_data_6_FT_minoxidil_balanced_with_exp \
    --val-data data/HKU_data_6_experiment \
    --pretrained checkpoints/best_model.pth
```

> **Note:** Due to stochasticity in training (random initialization, data shuffling), fine-tuning results may vary slightly across runs. The provided `best_FT_model.pth` is the exact model used to produce all results reported in the paper.

### Pre-training (requires external data download)

```bash
python scripts/train.py \
    --data data/HKU_data_5_total_inbalance \
    --epochs 400 --batch-size 64
```

### Full Retraining From NPZ Files (No CCDC)

If you have the precomputed NPZ feature files, full retraining does not require
CCDC:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

bash scripts/retrain_from_npz.sh
```

See `docs/retraining-no-ccdc.md` for required files, environment notes, and
cluster-friendly overrides.

## Model Architecture

MCC-GCN uses a graph convolutional network to learn from molecular pair graphs:

```
Input (34-dim atom features)
  → GCNConv(34, 256) + BN + ReLU
  → GCNConv(256, 256) + BN + ReLU
  → GCNConv(256, 128) + BN + ReLU
  → Global Mean Pooling
  → FC(128, 128) + BN + ReLU + Dropout(0.208)
  → FC(128, 64) + BN + ReLU + Dropout(0.208)
  → FC(64, 4) → Softmax
```

**Output classes:** 0 = Negative, 1 = Salt, 2 = Cocrystal, 3 = Solvate

**Bidirectional averaging:** During evaluation, each molecular pair is evaluated in both input orders (A-B and B-A), and the softmax probabilities are averaged before final classification.

## Project Structure

```
MCC-GCN/
├── mcc_gcn/                    # Core package
│   ├── featurize/              # Feature extraction (atom, bond, coformer, cocrystal)
│   ├── models/                 # GCN model, training/evaluation loops, metrics
│   ├── data/                   # Dataset classes
│   └── utils.py                # Utility functions
├── scripts/
│   ├── train.py                # Pre-training
│   ├── finetune.py             # Fine-tuning
│   ├── evaluate.py             # Evaluation with bidirectional averaging
│   └── predict.py              # Single-pair prediction (SMILES/CAS/SDF)
├── data/                       # Pre-computed features and datasets
├── checkpoints/                # Model weights
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{deng2026mccgcn,
  title={MCC-GCN: An Interpretable Graph Learning Framework for Multicomponent Crystal Classification and Discovery},
  author={Deng, Yuehua and Zhao, Fanyu and Zhou, Xinliang and Fu, Minqi and Chow, Stephanie and Wei, Zhi and Wen, Qingsong and Chow, Shing Fung},
  journal={Nature Communications},
  year={2026},
  note={Under review}
}

@software{mccgcn_code,
  author={Deng, Yuehua and Zhao, Fanyu and Zhou, Xinliang and Fu, Minqi and Chow, Stephanie and Wei, Zhi and Wen, Qingsong and Chow, Shing Fung},
  title={PolarSnowLeopard/MCC-GCN: v1.0.0},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.19612214},
  url={https://doi.org/10.5281/zenodo.19612214}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
