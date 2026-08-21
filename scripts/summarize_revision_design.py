#!/usr/bin/env python3
"""Summarize frozen three-API data splits and MCC-GCN parameter counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcc_gcn.models.gcn import GCNNet


TASKS = {
    "binary": {"num_classes": 2, "model_size": "small"},
    "four-class": {"num_classes": 4, "model_size": "large"},
}
CLASS_NAMES = {
    0: "negative",
    1: "salt",
    2: "cocrystal",
    3: "hydrate_or_solvate",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default="data/revision-three-api",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def model_parameter_rows() -> list[dict]:
    rows = []
    for task, config in TASKS.items():
        for setting in [0, 1, 2, 3]:
            model = GCNNet(**config)
            model.ft_setting(train_dense_layer=setting)
            total = sum(parameter.numel() for parameter in model.parameters())
            trainable = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            rows.append(
                {
                    "task": task,
                    "model_size": config["model_size"],
                    "classes": config["num_classes"],
                    "fine_tuning_setting": setting,
                    "fine_tuning_scope": {
                        0: "all_parameters",
                        1: "output_layer",
                        2: "last_two_dense_layers",
                        3: "last_three_dense_layers",
                    }[setting],
                    "total_parameters": total,
                    "trainable_parameters": trainable,
                    "trainable_fraction": trainable / total,
                }
            )
    return rows


def _label_counts(table: pd.DataFrame) -> dict[str, int]:
    counts = table["label_int"].value_counts().sort_index()
    unknown = set(counts.index).difference(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown labels in frozen data: {sorted(unknown)}")
    return {
        CLASS_NAMES[int(key)]: int(value)
        for key, value in counts.items()
    }


def _split_rows(data_root: Path) -> tuple[list[dict], dict]:
    paths = {
        "pretraining_all": data_root
        / "four_class_pretrain_excluding_target_apis.csv",
        "pretraining_train": data_root
        / "pretrain-split"
        / "train_physical_pairs.csv",
        "pretraining_validation": data_root
        / "pretrain-split"
        / "validation_physical_pairs.csv",
        "target_all": data_root / "four_class_target_physical_pairs.csv",
    }
    tables = {
        name: pd.read_csv(path, keep_default_na=False)
        for name, path in paths.items()
    }
    rows = []
    for name, table in tables.items():
        counts = _label_counts(table)
        row = {
            "split": name,
            "physical_pairs": len(table),
            "unique_pair_keys": int(table["pair_key"].nunique()),
        }
        row.update({f"class_{key}": value for key, value in counts.items()})
        rows.append(row)

    train_keys = set(tables["pretraining_train"]["pair_key"])
    validation_keys = set(tables["pretraining_validation"]["pair_key"])
    target_keys = set(tables["target_all"]["representative_pair_key"])
    source_keys = set(tables["pretraining_all"]["pair_key"])
    checks = {
        "pretraining_train_validation_pair_overlap": len(
            train_keys.intersection(validation_keys)
        ),
        "pretraining_target_pair_overlap": len(source_keys.intersection(target_keys)),
    }
    return rows, checks


def _target_api_rows(target: pd.DataFrame) -> list[dict]:
    rows = []
    api_names = sorted(
        {
            name
            for values in target["target_apis"].str.split(";")
            for name in values
            if name
        }
    )
    for api_name in api_names:
        subset = target.loc[
            target["target_apis"].str.split(";").map(
                lambda values: api_name in values
            )
        ]
        row = {
            "target_api": api_name,
            "physical_pairs": len(subset),
        }
        row.update(
            {
                f"class_{key}": value
                for key, value in _label_counts(subset).items()
            }
        )
        rows.append(row)
    return rows


def _write_markdown(output_dir: Path, summary: dict):
    lines = [
        "# Three-API revision design summary",
        "",
        "## Frozen datasets",
        "",
        "| Split | Physical pairs | Negative | Salt | Cocrystal | Hydrate/solvate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["data_splits"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["split"],
                    str(row["physical_pairs"]),
                    str(row.get("class_negative", 0)),
                    str(row.get("class_salt", 0)),
                    str(row.get("class_cocrystal", 0)),
                    str(row.get("class_hydrate_or_solvate", 0)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Target APIs",
            "",
            "| API | Physical pairs | Negative | Salt | Cocrystal | Hydrate/solvate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["target_apis"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["target_api"],
                    str(row["physical_pairs"]),
                    str(row.get("class_negative", 0)),
                    str(row.get("class_salt", 0)),
                    str(row.get("class_cocrystal", 0)),
                    str(row.get("class_hydrate_or_solvate", 0)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## MCC-GCN parameters",
            "",
            "| Task | Model | Fine-tuning scope | Total | Trainable | Fraction |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["model_parameters"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task"],
                    row["model_size"],
                    row["fine_tuning_scope"],
                    str(row["total_parameters"]),
                    str(row["trainable_parameters"]),
                    f"{100 * row['trainable_fraction']:.2f}%",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The formal three-API MCC-GCN runs use `all_parameters`. The submitted",
            "paper's historical fine-tuning description corresponds to",
            "`last_three_dense_layers`; the two settings must not be conflated.",
        ]
    )
    (output_dir / "revision_design_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (data_root / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    split_rows, leakage_checks = _split_rows(data_root)
    target = pd.read_csv(
        data_root / "four_class_target_physical_pairs.csv",
        keep_default_na=False,
    )
    parameter_rows = model_parameter_rows()
    summary = {
        "schema_version": "mcc-gcn-revision-design-summary-v1",
        "source_manifest_schema": manifest["schema_version"],
        "data_splits": split_rows,
        "target_apis": _target_api_rows(target),
        "leakage_checks": leakage_checks,
        "outer_cross_validation": manifest["outer_cross_validation"],
        "model_parameters": parameter_rows,
        "revision_fine_tuning_setting": 0,
        "historical_paper_fine_tuning_setting": 3,
    }
    if any(leakage_checks.values()):
        raise ValueError(f"Frozen split leakage detected: {leakage_checks}")
    (output_dir / "revision_design_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(split_rows).to_csv(
        output_dir / "data_split_counts.csv",
        index=False,
    )
    pd.DataFrame(summary["target_apis"]).to_csv(
        output_dir / "target_api_counts.csv",
        index=False,
    )
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / "model_parameter_counts.csv",
        index=False,
    )
    _write_markdown(output_dir, summary)
    print(f"Revision design summary ready: {output_dir}")


if __name__ == "__main__":
    main()
