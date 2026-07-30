#!/usr/bin/env python
"""Build leakage-locked binary and four-class physical-pair tables."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.curation import curate_pretraining_pairs
from mcc_gcn.data.quality import read_pair_table


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csd-neutralization-audit", required=True)
    parser.add_argument("--negative-pairs", required=True)
    parser.add_argument("--external-split-manifest", required=True)
    parser.add_argument("--finetune-lock-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csd = pd.read_csv(
        args.csd_neutralization_audit,
        keep_default_na=False,
        low_memory=False,
    )
    negative = read_pair_table(args.negative_pairs)
    external = pd.read_csv(
        args.external_split_manifest,
        keep_default_na=False,
    )
    finetune = pd.read_csv(
        args.finetune_lock_manifest,
        keep_default_na=False,
    )
    locked_pair_keys = set(external["pair_key"])
    locked_pair_keys.update(
        value for value in finetune["model_pair_key"] if value
    )
    locked_identifiers = {
        value for value in finetune["identifier"] if value
    }

    result = curate_pretraining_pairs(
        csd,
        negative,
        locked_pair_keys=locked_pair_keys,
        locked_identifiers=locked_identifiers,
    )
    outputs = {
        "pair_evidence.csv": result.evidence,
        "pair_conflicts.csv": result.conflicts,
        "four_class_physical_pairs.csv": result.four_class_pairs,
        "binary_physical_pairs.csv": result.binary_pairs,
    }
    output_hashes = {}
    for filename, table in outputs.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        output_hashes[filename] = sha256_file(path)

    status_counts = (
        result.evidence["curation_status"].value_counts().sort_index()
    )
    reason_counts = (
        result.evidence["curation_reasons"]
        .str.split(";")
        .explode()
        .loc[lambda values: values.ne("")]
        .value_counts()
        .sort_index()
    )
    four_counts = (
        result.four_class_pairs.groupby(["label_str", "label_int"])
        .size()
        .sort_index()
    )
    binary_counts = (
        result.binary_pairs.groupby(["label_str", "label_int"])
        .size()
        .sort_index()
    )
    manifest = {
        "schema_version": "mcc-gcn-curated-task-pairs-v1",
        "inputs": {
            name: {
                "path": str(Path(path)),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "csd_neutralization_audit": (
                    args.csd_neutralization_audit
                ),
                "negative_pairs": args.negative_pairs,
                "external_split_manifest": (
                    args.external_split_manifest
                ),
                "finetune_lock_manifest": args.finetune_lock_manifest,
            }.items()
        },
        "locked_pair_keys": len(locked_pair_keys),
        "locked_identifiers": len(locked_identifiers),
        "evidence_status_counts": {
            str(status): int(count)
            for status, count in status_counts.items()
        },
        "evidence_reason_counts": {
            str(reason): int(count)
            for reason, count in reason_counts.items()
        },
        "four_class_pair_counts": {
            f"{label}:{label_int}": int(count)
            for (label, label_int), count in four_counts.items()
        },
        "binary_pair_counts": {
            f"{label}:{label_int}": int(count)
            for (label, label_int), count in binary_counts.items()
        },
        "outputs": output_hashes,
    }
    manifest_path = output_dir / "curation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
