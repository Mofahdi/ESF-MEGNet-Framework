"""Resolve portable material locations without rewriting the target tables."""

from pathlib import Path


def material_directories(frame, data_root, material_root=None):
    if material_root is None:
        return [(Path(data_root) / str(value)).resolve() for value in frame.mat_dir]
    root = Path(material_root).resolve()
    directories = []
    for value in frame.id:
        name = str(value)
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"Material ID must be a single directory name: {name!r}")
        directory = (root / name).resolve()
        if directory.parent != root:
            raise ValueError(f"Material directory escapes material_root: {name!r}")
        directories.append(directory)
    return directories
