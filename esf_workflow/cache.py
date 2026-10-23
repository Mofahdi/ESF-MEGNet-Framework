"""Cache identity includes inputs, targets, preprocessing, and random-feature seed."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .splits import file_sha256


def dataset_cache_key(dataset):
    def describe(value):
        if isinstance(value, (list, tuple)):
            return [describe(item) for item in value]
        if value is None:
            return None
        path = Path(value).resolve()
        return {"path": str(path), "sha256": file_sha256(path)}

    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot fingerprint cache setting {value!r}")

    source_paths = [Path(inspect.getfile(type(dataset))), Path(inspect.getfile(type(dataset.converter)))]
    neighbor_source = source_paths[1].with_name("nbrs_map.py")
    if neighbor_source.is_file():
        source_paths.append(neighbor_source)
    contents = {
        "schema": 1,
        "inputs": {name: describe(value) for name, value in vars(dataset).items() if name.endswith("s_path")},
        "labels": dataset.labels, "ids": dataset.ids, "graph_labels": dataset.graph_labels,
        "converter": vars(dataset.converter), "torch_seed": torch.initial_seed(),
        "include_line_graph": dataset.include_line_graph,
        "threebody_cutoff": dataset.threebody_cutoff, "directed_line_graph": dataset.directed_line_graph,
        "sources": [describe(path) for path in source_paths],
    }
    return hashlib.sha256(json.dumps(contents, sort_keys=True, default=encode).encode()).hexdigest()
