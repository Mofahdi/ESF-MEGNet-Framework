#!/usr/bin/env python3
"""Create shared CV indices without copying or reordering the source CSV."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esf_workflow.splits import generate_cv_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-filename", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=10)
    args = parser.parse_args()
    metadata = generate_cv_splits(args.csv_filename, args.output_dir, args.seed, args.n_splits)
    print(f"Saved {metadata['n_splits']} disjoint folds for {metadata['n_samples']} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
