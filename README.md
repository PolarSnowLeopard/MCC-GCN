# MCC-GCN

**Multi-Component Crystal Graph Convolutional Network** — An interpretable graph learning framework for multicomponent crystal classification and discovery.

MCC-GCN extends cocrystal prediction from binary classification to **four-class crystal form prediction** (cocrystal, salt, solvate, negative), and provides **gradient-based interpretability** at atomic resolution.

## Highlights

- **Four-class classification**: Simultaneously distinguishes cocrystals, salts, solvates, and negative outcomes
- **Transfer learning**: Pretrained on 34,000+ CSD entries, fine-tuned with only 34 samples for novel APIs
- **No CCDC required**: Pre-computed features and trained weights are provided — prediction works with SMILES input only
- **Interpretable**: Gradient-based attribution identifies key functional groups driving crystal formation

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

> **Download link:** [TODO: Zenodo DOI — upload pending]
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

Expected output:

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

### Reproduce Paper Results (Table 2, four-class)

```bash
python scripts/evaluate.py \
    --model checkpoints/best_FT_model.pth \
    --test-data-1 data/HKU_data_6_experiment_1 \
    --test-data-2 data/HKU_data_6_experiment_2
```

Expected output:

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

These results correspond to the fine-tuned MCC-GCN row in **Table 2** and the confusion matrix in **Figure 3d** of the paper.

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
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
