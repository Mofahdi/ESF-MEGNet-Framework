# ESF-MEGNet

Materials property prediction with MEGNet and electronic structure features.
The three ESF variants share the same architecture and use real, zero, or random
electronic features; MEGNet is a separate architectural baseline.
Electronic inputs are Löwdin charge, ICOHP, and ICOBI; DOS features are not used.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
the following from the repository root. uv downloads Python 3.11 if needed.
This setup targets Linux with an NVIDIA GPU and a CUDA 12.1-compatible driver:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend cu121 -r requirements.txt
```

## Reproduce

From the repository root, with the environment activated:

```bash
python scripts/reproduce.py
```

No options are needed. Training uses one NVIDIA GPU per fit. The workflow covers
phonon DOS peak frequency, average MSD, bulk modulus, shear modulus, and Vickers
hardness with **seed 42**, four models, and
10 shared, disjoint cross-validation folds per property (**200 fits**).
Missing structure and LOBSTER files are downloaded automatically from
[NOMAD](https://doi.org/10.17172/nomad.mc38-9gm1) into `data/LOBSTER_data/`;
complete local materials are reused.

Defaults and target columns are in [configs/properties.json](configs/properties.json).
The main entry point calls [scripts/reproduce_target.py](scripts/reproduce_target.py)
for each property. Results are saved under `runs/all_properties_seed42/`, with
combined metrics in `summary_all_properties.csv`. Reruns create numbered output
directories and preserve earlier results.

This runs the corrected workflow; the manuscript's numerical results have not
yet been revalidated. Phonon targets now use the DFPT dataset instead of Togo's
database. The modulus outlier `mp-757233` still needs source verification.

Run software checks with `python -m unittest discover -s tests -v`.
