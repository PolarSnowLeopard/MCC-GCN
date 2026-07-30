#!/usr/bin/env python
"""Audit and mechanically curate one or more molecular-pair CSV files."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.quality import (
    DEFAULT_ALLOWED_ELEMENTS,
    audit_pair_table,
    build_clean_pair_table,
    build_conflict_table,
    build_source_overlap_table,
    read_pair_table,
    summarize_audit,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate SMILES, detect unordered-pair duplicates and label "
            "conflicts, and emit a conservative clean candidate table."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Pair CSV files. Headered and legacy headerless files are supported.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allowed-elements",
        default=",".join(sorted(DEFAULT_ALLOWED_ELEMENTS)),
        help="Comma-separated element symbols accepted by the curation step.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_elements = {
        value.strip()
        for value in args.allowed_elements.split(",")
        if value.strip()
    }

    tables = [read_pair_table(path) for path in args.inputs]
    combined = pd.concat(tables, ignore_index=True)
    audited = audit_pair_table(
        combined, allowed_elements=allowed_elements
    )

    summary = {
        "inputs": [str(Path(path)) for path in args.inputs],
        "allowed_elements": sorted(allowed_elements),
        "combined": summarize_audit(audited),
        "by_source": {
            source: summarize_audit(group)
            for source, group in audited.groupby("source_file", sort=True)
        },
    }
    overlap = build_source_overlap_table(audited)
    summary["source_pair_overlap"] = overlap.to_dict(orient="records")
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audited.to_csv(output_dir / "audited_rows.csv", index=False)
    audited[audited["issue_codes"].ne("")].to_csv(
        output_dir / "quarantined_rows.csv", index=False
    )
    build_conflict_table(audited).to_csv(
        output_dir / "conflicting_pairs.csv", index=False
    )
    build_clean_pair_table(audited).to_csv(
        output_dir / "clean_pair_candidates.csv", index=False
    )
    overlap.to_csv(output_dir / "source_pair_overlap.csv", index=False)

    print(json.dumps(summary["combined"], indent=2, sort_keys=True))
    print(f"Audit artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
