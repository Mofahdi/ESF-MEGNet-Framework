"""Generate and validate reproducible, disjoint train/validation/test splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_indices(train, val, test, n_samples):
    """Reject leakage, duplicates, invalid indices, and incomplete partitions."""
    partitions = []
    for name, values in zip(("train", "val", "test"), (train, val, test)):
        values = np.asarray(values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} indices must be a one-dimensional integer array")
        if not len(values) or np.any(values < 0) or np.any(values >= n_samples):
            raise ValueError(f"{name} indices are empty or outside [0, {n_samples})")
        members = set(values.tolist())
        if len(members) != len(values):
            raise ValueError(f"Duplicate indices in {name} split")
        partitions.append(members)
    for i, left in enumerate(("train", "val", "test")):
        for j in range(i + 1, 3):
            overlap = partitions[i] & partitions[j]
            if overlap:
                right = ("train", "val", "test")[j]
                raise ValueError(f"{left}/{right} overlap: {len(overlap)} samples")
    if len(set.union(*partitions)) != n_samples:
        raise ValueError("The splits must cover every CSV row exactly once")


def generate_cv_splits(csv_path, output_dir, seed=42, n_splits=10):
    csv_path, output_dir = Path(csv_path).resolve(), Path(output_dir).resolve()
    n_samples = len(pd.read_csv(csv_path))
    # Reserve 10% of the whole dataset for validation, not 10% of trainval.
    n_val = int(np.ceil(0.1 * n_samples))
    folds = []
    for trainval, test in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        np.arange(n_samples)
    ):
        train, val = train_test_split(trainval, test_size=n_val, random_state=seed)
        validate_indices(train, val, test, n_samples)
        folds.append((train, val, test))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Split directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "csv_path": str(csv_path), "csv_sha256": file_sha256(csv_path),
        "n_samples": n_samples, "n_splits": n_splits, "seed": seed,
        "validation_fraction": 0.1,
    }
    for fold, partitions in enumerate(folds):
        fold_dir = output_dir / str(fold)
        fold_dir.mkdir()
        checksums = {}
        for name, values in zip(("train", "val", "test"), partitions):
            filename = fold_dir / f"{name}_idx_{fold}.npy"
            np.save(filename, values)
            checksums[filename.name] = file_sha256(filename)
        (fold_dir / "split_manifest.json").write_text(json.dumps(
            {**metadata, "fold": fold, "index_sha256": checksums}, indent=2
        ) + "\n")
    (output_dir / "split_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_indices(split_dir, fold, csv_path, n_samples):
    split_dir = Path(split_dir)
    paths = []
    suffixes = set()
    for name in ("train", "val", "test"):
        if fold is None:
            matches = list(split_dir.glob(f"{name}_idx_*.npy"))
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} index file in {split_dir}; specify --fold")
            path = matches[0]
        else:
            path = split_dir / f"{name}_idx_{fold}.npy"
        suffixes.add(path.stem.split("_idx_", 1)[1])
        paths.append(path)
    if len(suffixes) != 1:
        raise ValueError("Train, validation, and test filenames refer to different folds")
    manifest_path = split_dir / "split_manifest.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata["csv_sha256"] != file_sha256(csv_path):
            raise ValueError("CSV contents/order differ from the CSV used to generate these splits")
        if str(metadata["fold"]) != next(iter(suffixes)):
            raise ValueError("Index filenames do not match the fold manifest")
        for path in paths:
            if metadata["index_sha256"].get(path.name) != file_sha256(path):
                raise ValueError(f"Index file differs from the split manifest: {path}")
    partitions = tuple(np.load(path, allow_pickle=False) for path in paths)
    validate_indices(*partitions, n_samples)
    return partitions
