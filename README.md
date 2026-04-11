# MCC-GCN

Multi-Component Crystal Graph Convolutional Network for crystal type prediction.

## Reproduction Guide

### Step 1: Create Environment

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
```

### Step 2: Install MCC-GCN

```bash
git clone <repo-url>
cd mcc-gcn
pip install -e .
```

### Step 3: Download Data

Download the data package from [TODO: Add download link] and extract into `data/`.

Required files for training/evaluation (**no CCDC needed**):

| File | Purpose |
|------|---------|
| `HKU_data_5_total_inbalance.npz` | Pre-training features |
| `HKU_data_6_FT_minoxidil_balanced_with_exp.npz` | Fine-tuning features |
| `HKU_data_6_experiment_1.npz` | Evaluation features (A-B order) |
| `HKU_data_6_experiment_2.npz` | Evaluation features (B-A order) |

Optional files (only needed with CCDC license for feature re-extraction):

| File | Purpose |
|------|---------|
| `HKU_data.pkl.gz` | Merged mol blocks |
| `CCDC_data.pkl.gz` | CSD mol blocks |
| `FT_data.pkl.gz` / `FT_data_extra.pkl.gz` | Fine-tuning mol blocks |
| `cas_to_smiles_dict.pkl` | CAS-to-SMILES lookup |

### Step 4: Run

> All commands below should be run from the `mcc-gcn/` directory.

**Pre-training:**

```bash
python scripts/train.py \
    --data data/HKU_data_5_total_inbalance \
    --epochs 400 --batch-size 64
```

**Fine-tuning:**

```bash
python scripts/finetune.py \
    --data data/HKU_data_6_FT_minoxidil_balanced_with_exp \
    --val-data data/HKU_data_6_experiment \
    --pretrained checkpoints/best_model.pth
```

**Evaluation:**

```bash
python scripts/evaluate.py \
    --model checkpoints/best_FT_model.pth \
    --test-data-1 data/HKU_data_6_experiment_1 \
    --test-data-2 data/HKU_data_6_experiment_2
```

**Prediction (single sample, no CCDC needed):**

```bash
# From SMILES
python scripts/predict.py \
    --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" "OC(=O)CC(=O)O" \
    --model checkpoints/best_FT_model.pth

# From CAS numbers
python scripts/predict.py \
    --cas "58-08-2" "141-82-2" \
    --model checkpoints/best_FT_model.pth

# From SDF files
python scripts/predict.py \
    --sdf mol1.sdf mol2.sdf \
    --model checkpoints/best_FT_model.pth
```

The scripts automatically load pre-computed `.npz` features when available. If `.npz` is absent but `.csv` and `HKU_data.pkl.gz` exist, features will be rebuilt (requires CCDC). If neither is available, a clear error message will tell you which files to download.

> **Note:** Training on CPU is supported but slow. For a quick test, use `--epochs 2`.

## Project Structure

```
mcc-gcn/
├── mcc_gcn/                    # Core package
│   ├── featurize/              # Feature extraction (atom, bond, coformer, cocrystal, etc.)
│   ├── models/                 # GCN model, training/evaluation loops, metrics
│   ├── data/                   # Dataset classes, data filtering
│   └── utils.py                # Utility functions
├── scripts/
│   ├── data/                   # Data preparation scripts (require CCDC)
│   ├── train.py                # Pre-training
│   ├── finetune.py             # Fine-tuning
│   ├── evaluate.py             # Evaluation
│   └── predict.py              # Single-sample prediction (no CCDC needed)
├── data/                       # Datasets, features, and mol blocks
├── pyproject.toml
└── requirements.txt
```

## Data Preparation (Optional, requires CCDC)

These scripts reproduce how the datasets were originally constructed from the Cambridge Structural Database. They are **not needed** if you use the provided data files.

```bash
python scripts/data/collect_csd_data.py
python scripts/data/prepare_experiment_data.py --input data/Experimental_64_250713.csv
python scripts/data/build_ft_dataset.py
python scripts/data/merge_datasets.py --output data/HKU_data.pkl.gz
python scripts/data/build_training_set.py \
    --input data/HKU_data_4_reactions_total.csv \
    --mol-blocks data/HKU_data.pkl.gz \
    --output data/HKU_data_5_merge_crystal_dataset.csv
```
