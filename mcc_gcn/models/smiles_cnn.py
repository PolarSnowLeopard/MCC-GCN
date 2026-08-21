"""Order-symmetrized dual-branch SMILES convolutional network."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, rdBase
from torch.utils.data import Dataset


TOKEN_PATTERN = re.compile(
    r"\[[^\]]+\]|Br|Cl|Si|Na|Ca|Li|Mg|Al|@@?|%\d{2}|\d|"
    r"[A-Za-z]|[=#\-+()\\/\.:~]"
)
SPECIAL_TOKENS = ("<pad>", "<unk>")


def tokenize_smiles(smiles: str) -> list[str]:
    """Tokenize a SMILES string without silently dropping characters."""
    smiles = str(smiles)
    tokens = TOKEN_PATTERN.findall(smiles)
    if "".join(tokens) != smiles:
        raise ValueError(f"Unsupported SMILES token in: {smiles}")
    return tokens


class SmilesTokenizer:
    def __init__(self, vocabulary: list[str], *, max_length: int = 128):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        ordered = list(SPECIAL_TOKENS)
        ordered.extend(
            token
            for token in vocabulary
            if token not in SPECIAL_TOKENS
        )
        if len(ordered) != len(set(ordered)):
            raise ValueError("Tokenizer vocabulary contains duplicates")
        self.vocabulary = ordered
        self.max_length = max_length
        self.token_to_index = {
            token: index for index, token in enumerate(ordered)
        }

    @classmethod
    def from_smiles(cls, smiles_values, *, max_length: int = 128):
        tokens = sorted(
            {
                token
                for smiles in smiles_values
                for token in tokenize_smiles(smiles)
            }
        )
        return cls(tokens, max_length=max_length)

    def encode(self, smiles: str) -> torch.Tensor:
        tokens = tokenize_smiles(smiles)
        if len(tokens) > self.max_length:
            raise ValueError(
                f"SMILES has {len(tokens)} tokens; maximum is "
                f"{self.max_length}: {smiles}"
            )
        unknown = self.token_to_index["<unk>"]
        values = [self.token_to_index.get(token, unknown) for token in tokens]
        values.extend([0] * (self.max_length - len(values)))
        return torch.tensor(values, dtype=torch.long)

    def save(self, path: str | Path):
        Path(path).write_text(
            json.dumps(
                {
                    "schema_version": "mcc-gcn-smiles-tokenizer-v1",
                    "max_length": self.max_length,
                    "vocabulary": self.vocabulary,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocabulary = [
            token
            for token in payload["vocabulary"]
            if token not in SPECIAL_TOKENS
        ]
        return cls(vocabulary, max_length=int(payload["max_length"]))


def randomized_root_smiles(smiles: str, root_atom: int) -> str:
    """Create a deterministic non-canonical SMILES rooted at one atom."""
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if not 0 <= root_atom < molecule.GetNumAtoms():
        raise ValueError("root_atom falls outside the molecule")
    return Chem.MolToSmiles(
        molecule,
        canonical=False,
        rootedAtAtom=root_atom,
        isomericSmiles=True,
    )


class SmilesVariantCache:
    """Precompute deterministic randomized SMILES variants."""

    def __init__(self, smiles_values, *, variants: int = 4, seed: int = 42):
        if variants < 1:
            raise ValueError("variants must be positive")
        self.values: dict[str, tuple[str, ...]] = {}
        for index, smiles in enumerate(sorted(set(smiles_values))):
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
            roots = rng.permutation(molecule.GetNumAtoms())[:variants]
            generated = [
                Chem.MolToSmiles(
                    molecule,
                    canonical=False,
                    rootedAtAtom=int(root),
                    isomericSmiles=True,
                )
                for root in roots
            ]
            canonical = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )
            unique = tuple(dict.fromkeys([canonical, *generated]))
            self.values[str(smiles)] = unique

    def get(self, smiles: str, *, epoch: int, row_index: int, side: int) -> str:
        variants = self.values[str(smiles)]
        index = (epoch + row_index * 2 + side) % len(variants)
        return variants[index]


def task_labels(table: pd.DataFrame, task: str) -> np.ndarray:
    labels = table["label_int"].to_numpy(dtype=np.int64)
    if task == "binary":
        return np.where(labels == 0, 0, 1).astype(np.int64)
    if task == "four-class":
        return labels
    raise ValueError(f"Unsupported task: {task}")


class SmilesPairTrainingDataset(Dataset):
    """Two order augmentations per physical pair with randomized SMILES."""

    def __init__(
        self,
        table: pd.DataFrame,
        tokenizer: SmilesTokenizer,
        *,
        task: str,
        variants: int = 4,
        seed: int = 42,
    ):
        self.table = table.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.labels = task_labels(self.table, task)
        self.reactant_a = self.table["reactant_A"].to_numpy(dtype=str)
        self.reactant_b = self.table["reactant_B"].to_numpy(dtype=str)
        self.epoch = 0
        self.cache = SmilesVariantCache(
            list(self.table["reactant_A"]) + list(self.table["reactant_B"]),
            variants=variants,
            seed=seed,
        )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.table) * 2

    def __getitem__(self, index):
        row_index = index // 2
        reverse = bool(index % 2)
        left = self.reactant_b[row_index] if reverse else self.reactant_a[row_index]
        right = self.reactant_a[row_index] if reverse else self.reactant_b[row_index]
        left_random = self.cache.get(
            left,
            epoch=self.epoch,
            row_index=row_index,
            side=0,
        )
        right_random = self.cache.get(
            right,
            epoch=self.epoch,
            row_index=row_index,
            side=1,
        )
        return (
            self.tokenizer.encode(left_random),
            self.tokenizer.encode(right_random),
            torch.tensor(self.labels[row_index], dtype=torch.long),
        )


class SmilesPairEvaluationDataset(Dataset):
    """Canonical A/B and B/A inputs for probability-averaged evaluation."""

    def __init__(self, table, tokenizer: SmilesTokenizer, *, task: str):
        self.table = table.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.labels = task_labels(self.table, task)
        self.reactant_a = self.table["reactant_A"].to_numpy(dtype=str)
        self.reactant_b = self.table["reactant_B"].to_numpy(dtype=str)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        left = self.tokenizer.encode(self.reactant_a[index])
        right = self.tokenizer.encode(self.reactant_b[index])
        return (
            left,
            right,
            right.clone(),
            left.clone(),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class SmilesEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        *,
        embedding_dim: int = 32,
        filters: tuple[int, int] = (128, 256),
        kernel_size: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=0,
        )
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, filters[0], kernel_size),
                nn.Conv1d(filters[0], filters[1], kernel_size),
            ]
        )

    def forward(self, tokens):
        values = self.embedding(tokens).transpose(1, 2)
        for convolution in self.convolutions:
            values = F.selu(convolution(values))
        return F.adaptive_max_pool1d(values, 1).squeeze(-1)


class DualSmilesCNN(nn.Module):
    """Independent molecular CNN branches followed by an interaction head."""

    def __init__(
        self,
        vocabulary_size: int,
        num_classes: int,
        *,
        embedding_dim: int = 32,
        filters: tuple[int, int] = (128, 256),
        dense_dims: tuple[int, int] = (512, 128),
        kernel_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_args = {
            "embedding_dim": embedding_dim,
            "filters": filters,
            "kernel_size": kernel_size,
        }
        self.left_encoder = SmilesEncoder(vocabulary_size, **encoder_args)
        self.right_encoder = SmilesEncoder(vocabulary_size, **encoder_args)
        self.classifier = nn.Sequential(
            nn.Linear(filters[-1] * 2, dense_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dims[0], dense_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dims[1], num_classes),
        )

    def forward(self, left, right):
        left_features = self.left_encoder(left)
        right_features = self.right_encoder(right)
        return self.classifier(
            torch.cat([left_features, right_features], dim=1)
        )

    def freeze_encoders(self):
        for encoder in (self.left_encoder, self.right_encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and not any(
            parameter.requires_grad
            for parameter in self.left_encoder.parameters()
        ):
            self.left_encoder.eval()
            self.right_encoder.eval()
        return self
