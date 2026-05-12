#!/usr/bin/env python3
"""Create and audit patient-level ACDC split manifests.

The ACDC preprocessing in this repo stores ED and ES phases as separate
`patientXXX_ED.npy` / `patientXXX_ES.npy` volumes. Splitting by volume leaks
patient identity across train and validation, so paper experiments should split
by the `patientXXX` identifier first.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PATIENT_RE = re.compile(r"^(patient\d+)_")


def volume_id(path: str | Path) -> str:
    return Path(path).stem


def patient_id_from_volume(volume_name: str) -> str:
    match = PATIENT_RE.match(volume_name)
    return match.group(1) if match else volume_name


def discover_volumes(data_dir: str | Path) -> list[str]:
    volumes_dir = Path(data_dir) / "volumes"
    return sorted(p.name for p in volumes_dir.glob("*.npy"))


def group_by_patient(volume_files: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for filename in volume_files:
        vid = volume_id(filename)
        grouped.setdefault(patient_id_from_volume(vid), []).append(vid)
    return {pid: sorted(vols) for pid, vols in sorted(grouped.items())}


def build_split_manifest(
    data_dir: str | Path,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    data_dir = Path(data_dir)
    grouped = group_by_patient(discover_volumes(data_dir))
    if not grouped:
        raise FileNotFoundError(f"No .npy volumes found under {data_dir / 'volumes'}")

    patient_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    split_at = int(len(patient_ids) * train_ratio)
    train_patients = sorted(patient_ids[:split_at])
    val_patients = sorted(patient_ids[split_at:])

    train_volumes = sorted(v for pid in train_patients for v in grouped[pid])
    val_volumes = sorted(v for pid in val_patients for v in grouped[pid])

    manifest = {
        "dataset": "ACDC",
        "schema_version": 1,
        "split_type": "patient",
        "split_level": "patient",
        "patient_id_rule": "prefix_before_first_underscore",
        "seed": seed,
        "train_ratio": train_ratio,
        "train_fraction": train_ratio,
        "source_data_dir": str(data_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_patients": len(patient_ids),
        "num_patients": len(patient_ids),
        "n_volumes": sum(len(v) for v in grouped.values()),
        "num_volumes": sum(len(v) for v in grouped.values()),
        "train_patients": train_patients,
        "val_patients": val_patients,
        "train_volumes": train_volumes,
        "val_volumes": val_volumes,
        "splits": {
            "train": {
                "patients": train_patients,
                "volumes": train_volumes,
            },
            "val": {
                "patients": val_patients,
                "volumes": val_volumes,
            },
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    required = {"train_patients", "val_patients", "train_volumes", "val_volumes"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"Split manifest missing keys: {sorted(missing)}")

    train_patients = set(manifest["train_patients"])
    val_patients = set(manifest["val_patients"])
    overlap = train_patients & val_patients
    if overlap:
        raise ValueError(f"Patient split overlap: {sorted(overlap)[:10]}")

    for key in ("train_volumes", "val_volumes"):
        bad = [
            vol for vol in manifest[key]
            if patient_id_from_volume(vol) not in train_patients | val_patients
        ]
        if bad:
            raise ValueError(f"{key} contains volumes with unknown patients: {bad[:10]}")

    train_volume_patients = {patient_id_from_volume(v) for v in manifest["train_volumes"]}
    val_volume_patients = {patient_id_from_volume(v) for v in manifest["val_volumes"]}
    volume_overlap = train_volume_patients & val_volume_patients
    if volume_overlap:
        raise ValueError(f"Volume-derived patient overlap: {sorted(volume_overlap)[:10]}")


def load_manifest(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest(manifest)
    return manifest


def save_manifest(manifest: dict, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an ACDC patient-level split manifest")
    parser.add_argument("--data_dir", default="preprocessed_data/ACDC")
    parser.add_argument("--output", default="splits/acdc_patient_split_seed42.json")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    manifest = build_split_manifest(args.data_dir, args.train_ratio, args.seed)
    print(
        "ACDC patient split: "
        f"{len(manifest['train_patients'])} train patients / "
        f"{len(manifest['val_patients'])} val patients; "
        f"{len(manifest['train_volumes'])} train volumes / "
        f"{len(manifest['val_volumes'])} val volumes"
    )
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    save_manifest(manifest, args.output)
    print(f"Saved split manifest: {args.output}")


if __name__ == "__main__":
    main()
