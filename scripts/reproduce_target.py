#!/usr/bin/env python3
"""Run shared CV folds for one target; the full experiment uses reproduce.py."""

import argparse
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from esf_workflow.settings import DEFAULT_SETTINGS, validate_settings
from esf_workflow.splits import generate_cv_splits
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


def run_logged(command, log_path, env):
    """Stream both child output channels to the terminal and a persistent log."""
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
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
    """Run one target with explicit settings, using the current Python interpreter."""
    settings = validate_settings({} if settings is None else settings)
    csv_path = Path(csv_filename).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Choose a new run directory; this path already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata = generate_cv_splits(csv_path, output_dir / "splits", settings["split_seed"], settings["n_splits"])
    print(f"Saved {metadata['n_splits']} disjoint folds for {metadata['n_samples']} rows to {output_dir / 'splits'}", flush=True)
    save_provenance(output_dir)

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
    for model in settings["models"]:
        for fold in range(settings["n_splits"]):
            fold_dir = output_dir / model / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
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
            (fold_dir / "command.json").write_text(json.dumps(invocation, indent=2) + "\n")
            print(f"Running {model}, fold {fold + 1}/{settings['n_splits']}...", flush=True)
            run_logged(command, fold_dir / "training.log", env)
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
    help="Read materials from PATH/<material-id>/",
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
                   material_root=os.environ.get("MATERIAL_ROOT") or None)
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode if error.returncode > 0 else 128 - error.returncode, f"{error}\n")
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted; completed fold outputs have been preserved.\n")


if __name__ == "__main__":
    main()
