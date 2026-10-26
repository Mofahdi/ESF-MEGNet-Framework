"""Validate saved folds and preserve partial output when restarting a run."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .splits import file_sha256, generate_cv_splits, load_indices


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def run_lock(directory):
    """Reject concurrent writers; a child may inherit the descriptor for crash safety."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is already using this run: {directory}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        # Closing releases the lock once every inherited child descriptor is closed.
        yield handle.fileno()


def archive_partial(path, run_dir):
    """Move incomplete output aside without deleting checkpoints or logs."""
    path, run_dir = Path(path), Path(run_dir)
    archive = run_dir / "_incomplete"
    archive.mkdir(exist_ok=True)
    name = "_".join(path.relative_to(run_dir).parts)
    number = 1
    destination = archive / f"{name}_{number}"
    while destination.exists():
        number += 1
        destination = archive / f"{name}_{number}"
    path.rename(destination)
    return destination


def prepare_splits(csv_path, run_dir, settings):
    """Reuse only verified splits; install newly generated folds atomically."""
    csv_path, run_dir = Path(csv_path), Path(run_dir)
    directory = run_dir / "splits"
    manifest = directory / "split_manifest.json"
    expected = {
        "csv_sha256": file_sha256(csv_path), "n_splits": settings["n_splits"],
        "seed": settings["split_seed"], "n_samples": len(pd.read_csv(csv_path)),
    }
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Saved splits do not match the current data/settings: {directory}")
        test_indices = []
        for fold in range(settings["n_splits"]):
            if not (directory / str(fold) / "split_manifest.json").is_file():
                raise ValueError(f"Missing split manifest for fold {fold}: {directory}")
            _, _, test = load_indices(directory / str(fold), fold, csv_path, expected["n_samples"])
            test_indices.extend(test.tolist())
        if sorted(test_indices) != list(range(expected["n_samples"])):
            raise ValueError("Saved test folds must cover every row exactly once")
        return metadata
    if any(run_dir.glob("*/fold_*/command.json")) or any(run_dir.glob("*/fold_*/results.json")):
        raise ValueError(f"Cannot resume fold outputs without their split manifest: {directory}")
    if directory.exists():
        archive_partial(directory, run_dir)
    with tempfile.TemporaryDirectory(prefix=".splits-", dir=run_dir) as temporary:
        staging = Path(temporary) / "splits"
        metadata = generate_cv_splits(csv_path, staging, settings["split_seed"], settings["n_splits"])
        staging.rename(directory)
    return metadata


def fold_complete(directory, model, fold, manifest, frame, col_name, settings):
    """Require full, consistent predictions and metadata, not just a checkpoint."""
    directory = Path(directory)
    try:
        result = json.loads((directory / "results.json").read_text())
        if any(result.get(key) != value for key, value in {
            "variant": model, "fold": fold, "csv_sha256": manifest["csv_sha256"],
            "seed": settings["seed"], "feature_seed": settings["feature_seed"],
        }.items()):
            return False
        if not all(np.isfinite(result[key]) for key in ("train_MAE_min", "val_MAE_min", "test_MAE")):
            return False
        if result["R2"] is not None and not np.isfinite(result["R2"]):
            return False
        metadata_path = Path(result["run_metadata"])
        metrics_path = Path(result["metrics_path"])
        if not all(path.resolve().is_relative_to(directory.resolve()) for path in (metadata_path, metrics_path)):
            return False
        metadata = json.loads(metadata_path.read_text())
        if not metrics_path.is_file() or not metrics_path.stat().st_size:
            return False
        arguments = metadata["arguments"]
        for key in ("seed", "feature_seed", "max_epochs", "patience", "batch_size", "decay_steps", "device"):
            if arguments[key] != settings[key]:
                return False
        if arguments["col_name"] != col_name or metadata["csv_sha256"] != manifest["csv_sha256"]:
            return False
        split_dir = directory.parent.parent / "splits" / str(fold)
        for name in ("train", "val", "test"):
            expected = np.load(split_dir / f"{name}_idx_{fold}.npy", allow_pickle=False)
            saved = np.load(metadata_path.parent / "splits" / f"{name}_idx.npy", allow_pickle=False)
            if not np.array_equal(saved, expected):
                return False
        predictions = pd.read_csv(directory / "results.csv")
        if not np.array_equal(predictions.row_index.to_numpy(), expected):
            return False
        if not np.array_equal(predictions.id.astype(str).to_numpy(), frame.iloc[expected].id.astype(str).to_numpy()):
            return False
        if not np.isfinite(predictions[["label", "preds_best"]].to_numpy(dtype=float)).all():
            return False
        if not np.allclose(predictions.label, frame.iloc[expected][col_name], rtol=1e-6, atol=1e-7):
            return False
        mae = np.mean(np.abs(predictions.label - predictions.preds_best))
        return bool(np.isclose(mae, result["test_MAE"], rtol=1e-6, atol=1e-7))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, EOFError):
        # An interrupted write or missing artifact makes the fold eligible for a fresh fit.
        return False
