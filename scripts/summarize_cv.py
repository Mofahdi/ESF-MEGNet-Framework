#!/usr/bin/env python3
"""Summarize complete CV runs, preserving predictions from each held-out fold."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(run_dir, models):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "splits/split_manifest.json").read_text())
    summaries = []
    for model in models:
        scores, predictions = [], []
        for fold in range(manifest["n_splits"]):
            directory = run_dir / model / f"fold_{fold}"
            result = json.loads((directory / "results.json").read_text())
            if result["csv_sha256"] != manifest["csv_sha256"] or result["fold"] != fold or result["variant"] != model:
                raise ValueError(f"Result metadata does not match this experiment: {directory}")
            expected = np.load(run_dir / "splits" / str(fold) / f"test_idx_{fold}.npy", allow_pickle=False)
            frame = pd.read_csv(directory / "results.csv")
            if not np.array_equal(frame.row_index.to_numpy(), expected):
                raise ValueError(f"Predictions do not match this fold's test indices: {directory}")
            frame["fold"] = fold
            predictions.append(frame)
            scores.append(result)
        combined = pd.concat(predictions, ignore_index=True).sort_values("row_index")
        if not np.array_equal(combined.row_index.to_numpy(), np.arange(manifest["n_samples"])):
            raise ValueError(f"Test folds must cover each sample exactly once: {model}")
        combined.to_csv(run_dir / f"{model}_out_of_fold.csv", index=False)
        row = {"model": model, "n_folds": len(scores)}
        for metric in ("test_MAE", "R2"):
            values = [result[metric] for result in scores]
            if any(value is None for value in values):
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
            else:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(run_dir / "summary.csv", index=False)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run_dir, args.models), indent=2))
