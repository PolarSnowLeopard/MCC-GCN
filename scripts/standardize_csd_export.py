#!/usr/bin/env python
"""Standardize raw CSD export shards without requiring a CCDC license."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

from rdkit import rdBase

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcc_gcn.data.standardize import standardize_exported_component


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Raw CSD .jsonl.gz shards.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for source_line, line in enumerate(handle, start=1):
                if line.strip():
                    yield path, source_line, json.loads(line)


def _join(values):
    return ";".join(str(value) for value in values)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components_path = output_dir / "standardized_components.csv"
    entries_path = output_dir / "standardized_entries.csv"
    rejects_path = output_dir / "standardization_rejects.csv"
    manifest_path = output_dir / "standardization_manifest.json"

    component_fields = [
        "identifier",
        "source_file",
        "source_line",
        "source_manifest_row",
        "source_label_str",
        "source_label_int",
        "component_index",
        "ccdc_smiles",
        "ccdc_inchi",
        "ccdc_inchi_key",
        "ccdc_formal_charge",
        "parse_source",
        "observed_smiles",
        "observed_inchi_key",
        "observed_formal_charge",
        "parent_smiles",
        "parent_inchi_key",
        "parent_formal_charge",
        "parent_changed",
        "fragment_count",
        "heavy_atom_count",
        "radical_electrons",
        "elements",
        "warnings",
    ]
    entry_fields = [
        "identifier",
        "source_file",
        "source_line",
        "source_manifest_row",
        "source_label_str",
        "source_label_int",
        "has_3d_structure",
        "has_disorder",
        "component_count",
        "standardized_component_count",
        "distinct_parent_count",
        "parent_inchi_keys",
        "warning_count",
    ]
    reject_fields = [
        "identifier",
        "source_file",
        "source_line",
        "component_index",
        "error",
    ]

    entry_count = 0
    component_count = 0
    rejected_components = 0
    warning_count = 0
    with open(  # noqa: SIM117 - Python 3.9 compatible context layout
        components_path, "w", newline="", encoding="utf-8"
    ) as comp_handle:
        with open(entries_path, "w", newline="", encoding="utf-8") as entry_handle:
            with open(rejects_path, "w", newline="", encoding="utf-8") as reject_handle:
                component_writer = csv.DictWriter(
                    comp_handle, fieldnames=component_fields
                )
                entry_writer = csv.DictWriter(
                    entry_handle, fieldnames=entry_fields
                )
                reject_writer = csv.DictWriter(
                    reject_handle, fieldnames=reject_fields
                )
                component_writer.writeheader()
                entry_writer.writeheader()
                reject_writer.writeheader()

                for source, source_line, record in read_records(args.inputs):
                    entry_count += 1
                    identifier = str(record.get("identifier", ""))
                    source_manifest = record.get("source_manifest", {})
                    successful = []
                    entry_warnings = 0
                    for component in record.get("components", []):
                        component_count += 1
                        result = standardize_exported_component(component)
                        if result.error:
                            rejected_components += 1
                            reject_writer.writerow(
                                {
                                    "identifier": identifier,
                                    "source_file": source,
                                    "source_line": source_line,
                                    "component_index": component.get(
                                        "component_index"
                                    ),
                                    "error": result.error,
                                }
                            )
                            continue

                        successful.append(result)
                        entry_warnings += len(result.warnings)
                        component_writer.writerow(
                            {
                                "identifier": identifier,
                                "source_file": source,
                                "source_line": source_line,
                                "source_manifest_row": source_manifest.get(
                                    "source_row", ""
                                ),
                                "source_label_str": source_manifest.get(
                                    "label_str", ""
                                ),
                                "source_label_int": source_manifest.get(
                                    "label_int", ""
                                ),
                                "component_index": component.get(
                                    "component_index"
                                ),
                                "ccdc_smiles": component.get("smiles", ""),
                                "ccdc_inchi": component.get("inchi", ""),
                                "ccdc_inchi_key": component.get(
                                    "inchi_key", ""
                                ),
                                "ccdc_formal_charge": component.get(
                                    "formal_charge", ""
                                ),
                                "parse_source": result.parse_source,
                                "observed_smiles": result.observed_smiles,
                                "observed_inchi_key": result.observed_inchi_key,
                                "observed_formal_charge": (
                                    result.observed_formal_charge
                                ),
                                "parent_smiles": result.parent_smiles,
                                "parent_inchi_key": result.parent_inchi_key,
                                "parent_formal_charge": (
                                    result.parent_formal_charge
                                ),
                                "parent_changed": result.parent_changed,
                                "fragment_count": result.fragment_count,
                                "heavy_atom_count": result.heavy_atom_count,
                                "radical_electrons": result.radical_electrons,
                                "elements": _join(result.elements),
                                "warnings": _join(result.warnings),
                            }
                        )

                    warning_count += entry_warnings
                    parent_keys = sorted(
                        {
                            result.parent_inchi_key
                            for result in successful
                            if result.parent_inchi_key
                        }
                    )
                    entry_writer.writerow(
                        {
                            "identifier": identifier,
                            "source_file": source,
                            "source_line": source_line,
                            "source_manifest_row": source_manifest.get(
                                "source_row", ""
                            ),
                            "source_label_str": source_manifest.get(
                                "label_str", ""
                            ),
                            "source_label_int": source_manifest.get(
                                "label_int", ""
                            ),
                            "has_3d_structure": record.get("entry", {}).get(
                                "has_3d_structure", ""
                            ),
                            "has_disorder": record.get("entry", {}).get(
                                "has_disorder", ""
                            ),
                            "component_count": len(
                                record.get("components", [])
                            ),
                            "standardized_component_count": len(successful),
                            "distinct_parent_count": len(parent_keys),
                            "parent_inchi_keys": _join(parent_keys),
                            "warning_count": entry_warnings,
                        }
                    )

    input_files = [
        {"path": str(Path(path)), "sha256": sha256_file(path)}
        for path in args.inputs
    ]
    manifest = {
        "schema_version": "mcc-gcn-standardized-components-v1",
        "rdkit_version": rdBase.rdkitVersion,
        "inputs": input_files,
        "entry_count": entry_count,
        "component_count": component_count,
        "rejected_component_count": rejected_components,
        "warning_count": warning_count,
        "outputs": {
            "standardized_components.csv": sha256_file(components_path),
            "standardized_entries.csv": sha256_file(entries_path),
            "standardization_rejects.csv": sha256_file(rejects_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if rejected_components:
        raise SystemExit(
            f"{rejected_components} components failed standardization; inspect "
            f"{rejects_path}"
        )


if __name__ == "__main__":
    main()
