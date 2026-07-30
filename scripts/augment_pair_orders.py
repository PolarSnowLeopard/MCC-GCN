#!/usr/bin/env python
"""Create A/B and B/A rows after a physical-pair split is frozen."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="Combined A/B and B/A rows.")
    parser.add_argument("--forward-output", help="A/B rows only.")
    parser.add_argument("--reverse-output", help="B/A rows only.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not any([args.output, args.forward_output, args.reverse_output]):
        raise ValueError(
            "At least one of --output, --forward-output, or "
            "--reverse-output is required"
        )
    table = pd.read_csv(args.input, keep_default_na=False)
    required = {
        "reactant_A",
        "reactant_B",
        "label_str",
        "label_int",
        "identifier",
        "pair_key",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    if table["pair_key"].nunique() != len(table):
        raise ValueError("Input must contain one row per physical pair")

    forward = table.copy()
    forward["pair_order"] = "A_B"
    reverse = table.copy()
    reverse[["reactant_A", "reactant_B"]] = reverse[
        ["reactant_B", "reactant_A"]
    ]
    reverse["pair_order"] = "B_A"
    augmented = pd.concat([forward, reverse], ignore_index=True)
    if len(augmented) != 2 * len(table):
        raise AssertionError("Pair-order augmentation count mismatch")
    if augmented.groupby("pair_key").size().ne(2).any():
        raise AssertionError("Every physical pair must have two ordered rows")

    outputs = []
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        augmented.to_csv(args.output, index=False)
        outputs.append(f"{len(augmented)} combined rows -> {args.output}")
    if args.forward_output:
        Path(args.forward_output).parent.mkdir(parents=True, exist_ok=True)
        forward.to_csv(args.forward_output, index=False)
        outputs.append(
            f"{len(forward)} forward rows -> {args.forward_output}"
        )
    if args.reverse_output:
        Path(args.reverse_output).parent.mkdir(parents=True, exist_ok=True)
        reverse.to_csv(args.reverse_output, index=False)
        outputs.append(
            f"{len(reverse)} reverse rows -> {args.reverse_output}"
        )
    print(
        f"Created pair orientations from {len(table)} physical pairs: "
        + "; ".join(outputs)
    )


if __name__ == "__main__":
    main()
