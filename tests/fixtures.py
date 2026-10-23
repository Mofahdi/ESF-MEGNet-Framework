"""Tiny synthetic structures and valid LOBSTER files, not scientific benchmarks."""

import gzip
from pathlib import Path

from ase import Atoms
from ase.io import write
import pandas as pd


def write_material(directory, distance=1.0, reverse_icobi=False, missing_icobi=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    structure = Atoms("CHO", positions=[[2, 2, 2], [2 + distance, 2, 2], [2, 3, 2]], cell=[10, 10, 10], pbc=True)
    write(directory / "CONTCAR.gz", structure, format="vasp")
    header = "label atom1 atom2 distance tx ty tz ICOXX\n"
    hp = header + f"1 C1 H2 {distance} 0 0 0 -10.0\n2 C1 O3 1.0 0 0 0 -20.0\n"
    pair = "H2 C1" if reverse_icobi else "C1 H2"
    bi = header + f"1 {pair} {distance} 0 0 0 0.1\n"
    if not missing_icobi:
        bi += "2 C1 O3 1.0 0 0 0 0.2\n"
    charge = "CHARGE\nSynthetic fixture\natom Mulliken Loewdin\n1 C 0.2 0.2\n2 H -0.3 -0.3\n3 O 0.1 0.1\n\nTotal 0 0\n"
    for name, text in (("ICOHPLIST.lobster.gz", hp), ("ICOBILIST.lobster.gz", bi), ("CHARGE.lobster.gz", charge)):
        with gzip.open(directory / name, "wt") as handle:
            handle.write(text)
    return directory


def write_dataset(directory, n_samples=20):
    directory = Path(directory)
    rows = []
    for index in range(n_samples):
        relative = f"materials/sample_{index}"
        write_material(directory / relative, distance=1 + index / 100, reverse_icobi=True)
        rows.append({"id": f"sample_{index}", "mat_dir": relative, "target": 0.1 + index / 20})
    csv = directory / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv
