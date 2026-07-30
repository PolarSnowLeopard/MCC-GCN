#!/usr/bin/env python
"""Export immutable CSD entry records from an identifier manifest.

This script must run in a licensed CCDC Python environment. It deliberately
does not neutralize, filter, relabel, or deduplicate structures.
"""

import argparse
import csv
import gzip
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "mcc-gcn-csd-raw-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV containing identifier and optional source label columns.",
    )
    parser.add_argument("--output", required=True, help="Output .jsonl.gz shard.")
    parser.add_argument(
        "--rejects",
        help="Rejected-entry JSONL path. Defaults beside --output.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional per-shard limit for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "identifier" not in reader.fieldnames:
            raise ValueError("Manifest must have an identifier column")
        rows = []
        for source_row, row in enumerate(reader, start=2):
            clean = {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
                if key is not None
            }
            if not clean.get("identifier"):
                raise ValueError(
                    f"Manifest row {source_row} has an empty identifier"
                )
            clean["source_row"] = source_row
            rows.append(clean)
    return rows


def select_shard(rows, shard_index, num_shards, limit=None):
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    selected = [
        row for index, row in enumerate(rows)
        if index % num_shards == shard_index
    ]
    return selected[:limit] if limit is not None else selected


def _safe_attribute(obj, name):
    try:
        value = getattr(obj, name)
        return value() if callable(value) else value
    except Exception:  # noqa: BLE001 - optional CCDC fields vary by release
        return None


def _inchi_fields(component, warnings):
    try:
        generated = component.generate_inchi()
        return generated.inchi, generated.key
    except Exception as exc:  # noqa: BLE001 - preserve component with warning
        warnings.append(
            {
                "field": "inchi",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return None, None


def _mol_block(component, warnings):
    try:
        return component.to_string("sdf")
    except Exception as exc:  # noqa: BLE001 - preserve component with warning
        warnings.append(
            {
                "field": "mol_block",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return None


def export_component(component, component_index):
    warnings = []
    inchi, inchi_key = _inchi_fields(component, warnings)
    atoms = list(component.atoms)
    return {
        "component_index": component_index,
        "smiles": _safe_attribute(component, "smiles"),
        "inchi": inchi,
        "inchi_key": inchi_key,
        "formula": _safe_attribute(component, "formula"),
        "formal_charge": _safe_attribute(component, "formal_charge"),
        "molecular_weight": _safe_attribute(component, "molecular_weight"),
        "atom_count": len(atoms),
        "elements": sorted(
            {
                str(_safe_attribute(atom, "atomic_symbol"))
                for atom in atoms
                if _safe_attribute(atom, "atomic_symbol") is not None
            }
        ),
        "mol_block": _mol_block(component, warnings),
        "warnings": warnings,
    }


def export_entry(csd_reader, manifest_row):
    identifier = manifest_row["identifier"]
    entry = csd_reader.entry(identifier)
    molecule = entry.molecule
    components = list(molecule.components)
    if not components:
        raise ValueError("CSD entry has no molecular components")

    return {
        "schema_version": SCHEMA_VERSION,
        "identifier": identifier,
        "source_manifest": manifest_row,
        "entry": {
            "has_3d_structure": _safe_attribute(entry, "has_3d_structure"),
            "has_disorder": _safe_attribute(entry, "has_disorder"),
            "chemical_name": _safe_attribute(entry, "chemical_name"),
            "synonyms": _safe_attribute(entry, "synonyms"),
        },
        "crystal": {
            "formula": _safe_attribute(molecule, "formula"),
            "formal_charge": _safe_attribute(molecule, "formal_charge"),
            "smiles": _safe_attribute(molecule, "smiles"),
            "component_count": len(components),
        },
        "components": [
            export_component(component, index)
            for index, component in enumerate(components)
        ],
    }


def _json_default(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return list(value)
    except TypeError:
        return str(value)


def write_json_line(handle, record):
    handle.write(
        json.dumps(
            record,
            ensure_ascii=True,
            sort_keys=True,
            default=_json_default,
        )
        + "\n"
    )


def main():
    args = parse_args()
    try:
        import ccdc
        from ccdc import io
    except ImportError as exc:
        raise SystemExit(
            "The CCDC Python API is unavailable. Run this script in the "
            "licensed CCDC environment."
        ) from exc

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    rejects_path = Path(args.rejects) if args.rejects else output_path.with_name(
        output_path.name.removesuffix(".jsonl.gz") + ".rejects.jsonl"
    )
    metadata_path = output_path.with_name(
        output_path.name.removesuffix(".jsonl.gz") + ".metadata.json"
    )
    for path in (output_path, rejects_path, metadata_path):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = read_manifest(manifest_path)
    rows = select_shard(
        all_rows,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        limit=args.limit,
    )

    started_at = datetime.now(timezone.utc)
    csd_reader = io.EntryReader("CSD")
    exported = 0
    rejected = 0
    with gzip.open(  # noqa: SIM117 - Python 3.9 compatible context layout
        output_path, "wt", encoding="utf-8", newline="\n"
    ) as output:
        with open(rejects_path, "w", encoding="utf-8", newline="\n") as rejects:
            for row in rows:
                try:
                    write_json_line(output, export_entry(csd_reader, row))
                    exported += 1
                except Exception as exc:  # noqa: BLE001 - per-entry rejection
                    write_json_line(
                        rejects,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "identifier": row["identifier"],
                            "source_manifest": row,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                    rejected += 1

    finished_at = datetime.now(timezone.utc)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "ccdc_version": getattr(ccdc, "__version__", None),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": len(all_rows),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_rows": len(rows),
        "exported_rows": exported,
        "rejected_rows": rejected,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "rejects": str(rejects_path),
        "rejects_sha256": sha256_file(rejects_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if rejected:
        raise SystemExit(
            f"Export completed with {rejected} rejected entries; inspect "
            f"{rejects_path}"
        )


if __name__ == "__main__":
    main()
