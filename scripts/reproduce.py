#!/usr/bin/env python3
"""Reproduce all five properties using configs/properties.json; no arguments needed."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esf_workflow.splits import file_sha256
from esf_workflow.paths import material_directories
from esf_workflow.settings import validate_settings
from scripts.reproduce_target import run_target


PROPERTY_TARGETS = {
    "phonon_dos_peak": ("linear", "cm^-1"),
    "average_msd": ("log10", "angstrom^2"),
    "bulk_modulus": ("log10", "GPa"),
    "shear_modulus": ("log10", "GPa"),
    "vickers_hardness": ("log10", "GPa"),
}


def read_plan(config_path, check_material_files=True):
    """Validate every property before launching any training or creating outputs."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or set(config) - {"settings", "properties", "material_root"}:
        raise ValueError("Configuration must contain only 'settings', 'properties' and optional 'material_root'")
    material_root = config.get("material_root")
    if material_root is not None:
        if not isinstance(material_root, str) or not material_root.strip():
            raise ValueError("material_root must be a nonempty path or null")
        material_root = (config_path.parent / material_root).resolve()
    settings = validate_settings(config.get("settings", {}))
    models = settings["models"]
    properties = config.get("properties")
    if not isinstance(properties, dict) or properties.keys() != PROPERTY_TARGETS.keys():
        raise ValueError(f"properties must contain exactly: {', '.join(PROPERTY_TARGETS)}")

    errors, resolved, datasets, missing_files = [], {}, {}, set()
    required_files = ["CONTCAR.gz", "ICOHPLIST.lobster.gz"]
    if any(model != "MEGNet" for model in models):
        required_files += ["ICOBILIST.lobster.gz", "CHARGE.lobster.gz"]
    for name, (scale, units) in PROPERTY_TARGETS.items():
        entry = properties[name]
        try:
            if not isinstance(entry, dict) or set(entry) - {"csv_filename", "col_name", "data_root", "target_scale"}:
                raise ValueError("expected csv_filename, col_name, target_scale and optional data_root")
            for key in ("csv_filename", "col_name"):
                if not isinstance(entry.get(key), str) or not entry[key].strip():
                    raise ValueError(f"fill in {key} with the actual dataset value")
            if entry.get("target_scale") != scale:
                raise ValueError(f"target_scale must be '{scale}'; the column must already use that scale")
            csv_path = (config_path.parent / entry["csv_filename"]).resolve()
            if csv_path not in datasets:
                datasets[csv_path] = (pd.read_csv(csv_path), file_sha256(csv_path))
            df, checksum = datasets[csv_path]
            columns = {"id", "mat_dir", entry["col_name"]}
            missing = columns - set(df.columns)
            if missing:
                raise ValueError(f"missing CSV columns: {sorted(missing)}")
            if df[list(columns)].isna().any().any() or df.id.duplicated().any():
                raise ValueError("IDs must be unique; IDs, material paths and targets must be nonmissing")
            targets = pd.to_numeric(df[entry["col_name"]], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(targets).all():
                raise ValueError("target values must be finite")
            n_samples = len(df)
            if n_samples < settings["n_splits"] or n_samples - math.ceil(n_samples / settings["n_splits"]) - math.ceil(0.1 * n_samples) < 1:
                raise ValueError("not enough samples for nonempty train/validation/test sets")
            base = entry.get("data_root")
            if base is not None and (not isinstance(base, str) or not base.strip()):
                raise ValueError("data_root must be a nonempty path or null")
            data_root = (config_path.parent / base).resolve() if base else csv_path.parent
            material_dirs = set(material_directories(df, data_root, material_root))
            absent = [directory / filename for directory in sorted(material_dirs)
                      for filename in required_files if not (directory / filename).is_file()]
            missing_files.update(map(str, absent))
            if absent and check_material_files:
                examples = ", ".join(str(path) for path in absent[:3])
                raise ValueError(f"{len(absent)} required input files are missing; examples: {examples}")
            resolved[name] = {
                "csv_filename": str(csv_path), "csv_sha256": checksum,
                "col_name": entry["col_name"], "target_scale": scale, "target_units": units,
                "data_root": str(data_root), "n_samples": n_samples,
                "material_root": str(material_root) if material_root else None,
            }
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise ValueError("Configuration/input checks failed:\n  " + "\n  ".join(errors))
    return {
        "config_path": str(config_path), "config_sha256": file_sha256(config_path),
        "settings": settings, "properties": resolved,
        "missing_material_files": sorted(missing_files),
        "total_fits": len(resolved) * len(models) * settings["n_splits"],
    }


def combine_summaries(output_dir, plan):
    frames = []
    for name, entry in plan["properties"].items():
        frame = pd.read_csv(output_dir / name / "summary.csv")
        models = plan["settings"]["models"]
        if len(frame) != len(models) or set(frame.model) != set(models):
            raise ValueError(f"Incomplete model summary for {name}")
        if not (frame.n_folds == plan["settings"]["n_splits"]).all():
            raise ValueError(f"Incomplete fold summary for {name}")
        frame.insert(0, "property", name)
        frame.insert(1, "target_column", entry["col_name"])
        frame.insert(2, "target_scale", entry["target_scale"])
        frame.insert(3, "target_units", entry["target_units"])
        frames.append(frame)
    destination = output_dir / "summary_all_properties.csv"
    pd.concat(frames, ignore_index=True).to_csv(destination, index=False)
    return destination


def create_run_directory(seed):
    """Reserve a fresh default directory, preserving previous runs on every invocation."""
    base = ROOT / "runs" / f"all_properties_seed{seed}"
    base.parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        candidate = base if number == 1 else base.with_name(f"{base.name}_{number}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            number += 1


def run_all(config_path=None, output_dir=None, check_only=False):
    """Prepare missing inputs automatically, then run the configured experiment.

    Explicit paths and check_only are available to Python callers and tests;
    the command-line entry point uses the repository defaults.
    """
    config_path = Path(config_path or ROOT / "configs/properties.json").resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else None
    if not check_only and output_dir is not None and output_dir.exists():
        raise FileExistsError(f"Choose a new output directory; this path exists: {output_dir}")
    # Check every configuration value and target before downloading or creating outputs.
    plan = read_plan(config_path, check_material_files=False)
    material_root = next(iter(plan["properties"].values()))["material_root"]
    managed_data = material_root and (Path(material_root) / "download_manifest.json").is_file()
    if material_root and (plan["missing_material_files"] or managed_data):
        print("Checking NOMAD inputs and preparing missing or damaged files...", flush=True)
        from esf_workflow.download import download_for_config
        download_for_config(config_path)
    if plan["missing_material_files"]:
        # A failed or incomplete preparation must never start a partial experiment.
        plan = read_plan(config_path)
    else:
        print("All required material files are available locally; using existing data.", flush=True)
    for name, entry in plan["properties"].items():
        print(f"{name}: {entry['n_samples']} samples, column={entry['col_name']!r}, scale={entry['target_scale']}", flush=True)
    print(f"Validated five properties; {plan['total_fits']} training fits configured.", flush=True)
    if check_only:
        return plan

    if output_dir is None:
        output_dir = create_run_directory(plan["settings"]["seed"])
    else:
        output_dir.mkdir(parents=True)
    print(f"Results directory: {output_dir}", flush=True)
    state = {
        **plan, "output_dir": str(output_dir), "status": "running",
        "property_status": {name: "pending" for name in plan["properties"]},
    }
    def save_state():
        temporary = output_dir / "run_all.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output_dir / "run_all.json")

    save_state()
    current = None
    try:
        for name, entry in plan["properties"].items():
            current = name
            state["property_status"][name] = "running"
            save_state()
            if file_sha256(entry["csv_filename"]) != entry["csv_sha256"]:
                raise ValueError(f"{name}: CSV changed after the input checks")
            print(f"\nRunning {name}...", flush=True)
            run_target(
                entry["csv_filename"], entry["col_name"], output_dir / name,
                settings=plan["settings"], data_root=entry["data_root"],
                material_root=entry["material_root"],
            )
            state["property_status"][name] = "complete"
            save_state()
        current = None
        summary = combine_summaries(output_dir, plan)
        state["status"] = "complete"
        save_state()
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        if current:
            state["property_status"][current] = state["status"]
        state["error"] = str(error)
        save_state()
        raise
    print(f"Completed all five properties. Combined summary: {summary}", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Missing data are downloaded automatically. Settings live in configs/properties.json; results go under runs/.",
    )
    parser.parse_args()
    try:
        run_all()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed property outputs have been preserved.\n")


if __name__ == "__main__":
    main()
