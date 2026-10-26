#!/usr/bin/env python3
"""Run shared CV folds for one target; the full experiment uses reproduce.py."""

import argparse
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esf_workflow.settings import DEFAULT_SETTINGS, validate_settings
from esf_workflow.splits import file_sha256
from esf_workflow.resume import (
    archive_partial, command_differences, fold_complete, prepare_splits,
    run_lock, target_differences, write_json,
)
from scripts.summarize_cv import summarize


def save_provenance(output_dir):
    packages = {f"{dist.metadata['Name']}=={dist.version}" for dist in distributions()}
    (output_dir / "environment.txt").write_text("\n".join(sorted(packages, key=str.lower)) + "\n")
    git = ["git", "-C", str(ROOT)]
    try:
        available = subprocess.run(git + ["rev-parse", "--is-inside-work-tree"], capture_output=True)
    except FileNotFoundError:
        available = None
    if available is None or available.returncode != 0:
        (output_dir / "git_commit.txt").write_text("Source archive: Git metadata unavailable.\n")
        return
    for filename, arguments in (
        ("git_commit.txt", ["rev-parse", "HEAD"]),
        ("source_changes.patch", ["diff", "--binary", "HEAD"]),
        ("git_status.txt", ["status", "--short"]),
    ):
        with (output_dir / filename).open("wb") as handle:
            subprocess.run(git + arguments, stdout=handle, check=True)


def run_logged(command, log_path, env, lock_fd=None):
    """Stream both child output channels to the terminal and a persistent log."""
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            pass_fds=() if lock_fd is None else (lock_fd,),
        ) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                returncode = process.wait()
            except BaseException:
                # Do not leave a long training job running after an interrupted launcher.
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def run_target(csv_filename, col_name, output_dir, settings=None, data_root=None, material_root=None):
    """Reuse complete folds and restart incomplete fits with the same experiment settings."""
    settings = validate_settings({} if settings is None else settings)
    csv_path = Path(csv_filename).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    with run_lock(output_dir) as lock_fd:
        return _run_target(csv_path, col_name, output_dir, settings, data_root, material_root, lock_fd)


def _run_target(csv_path, col_name, output_dir, settings, data_root, material_root, lock_fd):
    expected = {
        "csv_filename": str(csv_path), "csv_sha256": file_sha256(csv_path),
        "col_name": col_name, "settings": settings,
        "data_root": str(Path(data_root).resolve()) if data_root else str(csv_path.parent),
        "material_root": str(Path(material_root).resolve()) if material_root else None,
    }
    manifest_path = output_dir / "target_run.json"
    frame = pd.read_csv(csv_path)
    if col_name not in frame or not {"id", "mat_dir"}.issubset(frame.columns):
        raise ValueError("CSV must contain IDs, material paths and the requested target column")
    path_cache = {}
    if manifest_path.exists():
        differences = target_differences(json.loads(manifest_path.read_text()), expected, frame, path_cache)
        if differences:
            raise ValueError(f"Existing target run has different data/settings: {output_dir}\n  "
                             + "\n  ".join(differences))

    deterministic_env = {
        "PYTHONHASHSEED": str(settings["seed"]), "DGLBACKEND": "pytorch",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
    }
    # These variables must be set before each fresh Python process imports torch/DGL.
    env = {**os.environ, **deterministic_env}
    data_args = []
    for flag, value in (("--data-root", data_root), ("--material-root", material_root)):
        if value is not None:
            data_args.extend([flag, str(Path(value).resolve())])
    jobs = []
    for model in settings["models"]:
        for fold in range(settings["n_splits"]):
            fold_dir = output_dir / model / f"fold_{fold}"
            command = [
                sys.executable, "-u", str(ROOT / model / "main.py"),
                "--csv-filename", str(csv_path), "--col-name", col_name,
                "--split-dir", str(output_dir / "splits" / str(fold)), "--fold", str(fold),
                "--output-dir", str(fold_dir),
            ]
            for key in ("seed", "feature_seed", "max_epochs", "patience", "batch_size", "decay_steps", "device"):
                command.extend(["--" + key.replace("_", "-"), str(settings[key])])
            command.extend(data_args)
            invocation = {"argv": command, "cwd": str(ROOT), "environment": deterministic_env}
            saved_command = fold_dir / "command.json"
            # This also validates runs created before target_run.json was introduced.
            if saved_command.exists():
                try:
                    previous = json.loads(saved_command.read_text())
                except (ValueError, OSError):
                    if not manifest_path.exists():
                        raise ValueError(f"Cannot verify the interrupted command: {saved_command}")
                else:
                    differences = command_differences(previous, invocation, frame, path_cache)
                    if differences:
                        raise ValueError(f"Saved fold command differs from the requested experiment: {saved_command}\n  "
                                         + "\n  ".join(differences))
            jobs.append((model, fold, fold_dir, invocation))
    expected_dirs = {directory for _, _, directory, _ in jobs}
    if any(directory not in expected_dirs for directory in output_dir.glob("*/fold_*")):
        raise ValueError(f"Saved folds do not match the requested models/fold count: {output_dir}")
    metadata = prepare_splits(csv_path, output_dir, settings)
    complete = [(manifest_path.exists() or (directory / "command.json").is_file())
                and fold_complete(directory, model, fold, metadata, frame, col_name, settings)
                for model, fold, directory, _ in jobs]
    if not manifest_path.exists():
        write_json(manifest_path, expected)
    if not (output_dir / "environment.txt").exists():
        save_provenance(output_dir)
    elif not all(complete):
        history = output_dir / "_resumes"
        history.mkdir(exist_ok=True)
        attempt = history / f"attempt_{len(list(history.iterdir())) + 1}"
        attempt.mkdir()
        save_provenance(attempt)
    print(f"{sum(complete)}/{len(jobs)} folds already complete; using verified shared splits.", flush=True)
    for (model, fold, fold_dir, invocation), done in zip(jobs, complete):
        label = f"{model}, fold {fold + 1}/{settings['n_splits']}"
        if done:
            print(f"Skipping completed {label}.", flush=True)
            continue
        if fold_dir.exists():
            preserved = archive_partial(fold_dir, output_dir)
            print(f"Restarting {label}; previous partial output saved to {preserved}.", flush=True)
        fold_dir.mkdir(parents=True)
        write_json(fold_dir / "command.json", invocation)
        print(f"Running {label}...", flush=True)
        started = time.monotonic()
        run_logged(invocation["argv"], fold_dir / "training.log", env, lock_fd=lock_fd)
        if not fold_complete(fold_dir, model, fold, metadata, frame, col_name, settings):
            raise RuntimeError(f"Training exited without complete, valid fold results: {fold_dir}")
        elapsed = time.monotonic() - started
        write_json(fold_dir / "timing.json", {"elapsed_seconds": elapsed})
        print(f"Completed {label} in {elapsed / 60:.1f} min.", flush=True)
    summary = summarize(output_dir, settings["models"])
    print(f"Completed. Fold means and standard deviations: {output_dir / 'summary.csv'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_filename")
    parser.add_argument("col_name")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--material-root",
        type=Path,
        default=None,
        metavar="PATH",
        help="Read materials from PATH/<material-id>/ (overrides MATERIAL_ROOT)",
    )
    args = parser.parse_args()
    try:
        # Retain environment overrides for the optional single-property entry point.
        supplied = {}
        for name, default in DEFAULT_SETTINGS.items():
            value = os.environ.get(name.upper())
            if value:
                if name == "models":
                    supplied[name] = value.split()
                elif type(default) is int:
                    supplied[name] = int(value)
                else:
                    supplied[name] = value
        run_target(args.csv_filename, args.col_name, args.output_dir, settings=supplied,
                   data_root=os.environ.get("DATA_ROOT") or None,
                   material_root=args.material_root or os.environ.get("MATERIAL_ROOT") or None)
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode if error.returncode > 0 else 128 - error.returncode, f"{error}\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed fold outputs have been preserved.\n")


if __name__ == "__main__":
    main()
