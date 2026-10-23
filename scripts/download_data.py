#!/usr/bin/env python3
"""Download the four required NOMAD files for all materials in a property config."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esf_workflow.download import download_for_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/properties.json")
    parser.add_argument("--output-dir", help="Default: material_root from the configuration")
    parser.add_argument("--mapping", help="Default: data/nomad_entry_ids.csv in the repository")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    try:
        download_for_config(args.config, args.output_dir, args.mapping, args.batch_size, args.workers)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
