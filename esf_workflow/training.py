"""One training/evaluation procedure shared by all four model variants."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import dgl
from dgl.data.utils import Subset
from dgl.dataloading import GraphDataLoader
import lightning as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch

from .splits import file_sha256, load_indices, validate_indices
from .paths import material_directories
from .settings import DEFAULT_SETTINGS


MODEL_CONFIG = dict(
    dim_node_embedding=118, dim_edge_embedding=50, dim_state_embedding=2,
    ntypes_state=None, nblocks=3, hidden_layer_sizes_input=(64, 32),
    hidden_layer_sizes_conv=(64, 64, 32), hidden_layer_sizes_output=(32, 16),
    nlayers_set2set=1, niters_set2set=2, activation_type="softplus2",
    is_classification=False, include_state=True, dropout=0.0,
    bond_expansion=None, cutoff=5.0, gauss_width=0.5,
)


class EpochProgress(Callback):
    """Plain progress lines that remain readable in terminals and captured logs."""

    def on_fit_start(self, trainer, pl_module):
        self.started = time.monotonic()
        self.last_epoch = self.last_reported = 0

    def report(self, trainer):
        metrics = trainer.callback_metrics
        parts = [f"Epoch {self.last_epoch}/{trainer.max_epochs}"]
        for key, label in (("train_MAE", "train MAE"), ("val_MAE", "val MAE"), ("val_RMSE", "val RMSE")):
            if key in metrics:
                parts.append(f"{label}={float(metrics[key]):.5g}")
        parts.append(f"elapsed={time.monotonic() - self.started:.1f}s")
        print(" | ".join(parts), flush=True)
        self.last_reported = self.last_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        self.last_epoch = trainer.current_epoch + 1
        if self.last_epoch == 1 or self.last_epoch % 10 == 0 or self.last_epoch == trainer.max_epochs:
            self.report(trainer)

    def on_fit_end(self, trainer, pl_module):
        if self.last_epoch != self.last_reported:
            self.report(trainer)


def seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    dgl.seed(seed)
    dgl.random.seed(seed)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_best_checkpoint(module, callback):
    """Use this fit's actual checkpoint, including Lightning's version suffix."""
    path = callback.best_model_path
    if not path or not Path(path).is_file():
        raise RuntimeError("Training produced no best checkpoint; evaluation cannot proceed")
    # Only checkpoints created locally by this training run are loaded here.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    module.cpu()
    module.load_state_dict(checkpoint["state_dict"])
    module.eval()
    return str(Path(path).resolve())


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-filename", required=True)
    parser.add_argument("--col-name", required=True)
    parser.add_argument("--data-root", help="Base for relative mat_dir paths; default: CSV directory")
    parser.add_argument("--material-root", help="Read MATERIAL_ROOT/<id>/ instead of the CSV's mat_dir paths")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--best-ckpt-dir", help="Default: OUTPUT_DIR/best_megCkpts")
    parser.add_argument("--best-ckpt-name", default="best")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=DEFAULT_SETTINGS["device"], help="Lightning accelerator: cuda (default), cpu, or auto")
    parser.add_argument("--seed", type=int, default=42, help="Model initialization and training seed")
    parser.add_argument("--feature-seed", type=int, default=42, help="Random input feature seed")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for --no-split-idx-avail")
    parser.add_argument("--decay-steps", type=int, default=1000, help="Cosine T_max in epochs")
    parser.add_argument("--split-dir", default=".")
    parser.add_argument("--fold", type=int, help="Explicit suffix of the index filenames")
    parser.add_argument("--split-idx-avail", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run_training(variant, model_class, dataset_class, converter_class, module_class, collate_fn, argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    if min(args.max_epochs, args.batch_size, args.decay_steps) < 1 or args.patience < 0:
        parser.error("epochs, batch size and decay steps must be positive; patience must be nonnegative")
    for name in ("seed", "feature_seed", "split_seed"):
        if not 0 <= getattr(args, name) < 2**32:
            parser.error(f"{name} must be in [0, 2**32)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    csv_path = Path(args.csv_filename).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else csv_path.parent
    df = pd.read_csv(csv_path)
    required = {"id", "mat_dir", args.col_name}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV is missing columns: {sorted(required - set(df.columns))}")
    if df[list(required)].isna().any().any() or df.id.duplicated().any():
        raise ValueError("CSV requires nonmissing IDs, paths and targets, and unique IDs")
    targets = pd.to_numeric(df[args.col_name], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("Targets must be finite")
    if args.split_idx_avail:
        partitions = load_indices(args.split_dir, args.fold, csv_path, len(df))
    else:
        trainval, test = train_test_split(np.arange(len(df)), test_size=0.1, random_state=args.split_seed)
        train, val = train_test_split(trainval, test_size=int(np.ceil(0.1 * len(df))), random_state=args.split_seed)
        partitions = train, val, test
        validate_indices(*partitions, len(df))

    # Resolve data paths before entering any output directory. Never copy/reorder the CSV.
    material_dirs = material_directories(df, data_root, args.material_root)
    def paths(filename):
        return [str(directory / filename) for directory in material_dirs]

    seed_everything(args.feature_seed)
    dataset_args = dict(
        name=f"{variant}_dataset", structures_path=paths("CONTCAR.gz"),
        labels={"target": targets.tolist()}, ids=df.id.tolist(),
        converter=converter_class(bond_length_cutoff=5, node_feats_type="Z"),
        save_cache=False, save_dir=str(output_dir),
    )
    if variant == "MEGNet":
        dataset_args["icoxxlists_path"] = paths("ICOHPLIST.lobster.gz")
    else:
        dataset_args.update(
            icohplists_path=paths("ICOHPLIST.lobster.gz"),
            icobilists_path=paths("ICOBILIST.lobster.gz"),
            chgcars_path=paths("CHARGE.lobster.gz"),
            doscars_path=[None] * len(df),  # DOS is not a model input in this implementation.
        )
    print(f"Preparing graphs for {len(df)} materials...", flush=True)
    preparation_started = time.monotonic()
    dataset = dataset_class(**dataset_args)
    print(f"Graphs ready in {time.monotonic() - preparation_started:.1f}s.", flush=True)
    train_data, val_data, test_data = [Subset(dataset, values.tolist()) for values in partitions]

    # Random control features must not consume the initialization/training RNG stream.
    seed_everything(args.seed)
    model = model_class(**MODEL_CONFIG)
    initial_parameters = parameter_digest(model)
    module = module_class(model=model, loss_params=None, loss="MAE", decay_steps=args.decay_steps)
    loader_options = dict(collate_fn=collate_fn, num_workers=0, batch_size=args.batch_size)
    train_loader = GraphDataLoader(train_data, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_options)
    val_loader = GraphDataLoader(val_data, shuffle=False, **loader_options)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(args.best_ckpt_dir).resolve()) if args.best_ckpt_dir else str(output_dir / "best_megCkpts"),
        monitor="val_MAE", filename=args.best_ckpt_name, save_top_k=1, mode="min",
    )
    logger = CSVLogger(str(output_dir / "logs"), name="MEGNet_training")
    log_dir = Path(logger.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    split_archive = log_dir / "splits"
    split_archive.mkdir()
    for name, values in zip(("train", "val", "test"), partitions):
        np.save(split_archive / f"{name}_idx.npy", values)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True)
        git_commit = revision.stdout.strip() if revision.returncode == 0 else None
    except FileNotFoundError:
        git_commit = None
    metadata = {
        "variant": variant, "arguments": vars(args), "csv_sha256": file_sha256(csv_path),
        "csv_path": str(csv_path), "data_root": str(data_root), "model_config": MODEL_CONFIG,
        "material_root": str(Path(args.material_root).resolve()) if args.material_root else None,
        "initial_parameters_sha256": initial_parameters,
        "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "split_sizes": dict(zip(("train", "val", "test"), map(len, partitions))),
        "git_commit": git_commit,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in (
            "torch", "dgl", "lightning", "torchmetrics", "numpy", "pandas", "scipy",
            "scikit-learn", "ase", "pymatgen", "matplotlib",
        )},
    }
    (log_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, accelerator=args.device, devices=1,
        deterministic=True, benchmark=False, logger=logger,
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_MAE", patience=args.patience, mode="min"), EpochProgress()],
        log_every_n_steps=1,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_path = load_best_checkpoint(module, checkpoint_callback)
    logger.save()
    metrics_path = log_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    fig, ax = plt.subplots()
    for name in ("train_MAE", "val_MAE"):
        values = metrics.dropna(subset=[name])
        ax.plot(values.epoch, values[name], label=name)
    ax.set(xlabel="Epoch", ylabel="MAE")
    ax.legend()
    for directory in (log_dir, output_dir):
        fig.savefig(directory / "train_val.jpg")
    plt.close(fig)

    ids, labels, predictions = [], [], []
    test_loader = GraphDataLoader(test_data, shuffle=False, collate_fn=collate_fn, batch_size=1, num_workers=0)
    with torch.inference_mode():
        for sample_ids, graph, lattice, state, target in test_loader:
            pred = module(graph, state_attr=state) * module.data_std + module.data_mean
            ids.extend(sample_ids)
            labels.extend(target.reshape(-1).tolist())
            predictions.extend(pred.reshape(-1).tolist())
    results = {
        "train_MAE_min": float(metrics.train_MAE.min()),
        "val_MAE_min": float(metrics.val_MAE.min()),
        "test_MAE": float(mean_absolute_error(labels, predictions)),
        "R2": float(r2_score(labels, predictions)) if len(labels) > 1 else None,
        "best_checkpoint": best_path, "metrics_path": str(metrics_path),
        "run_metadata": str(log_dir / "run.json"), "fold": args.fold,
        "seed": args.seed, "feature_seed": args.feature_seed, "variant": variant,
        "csv_sha256": metadata["csv_sha256"],
    }
    predictions_df = pd.DataFrame({"row_index": partitions[2], "id": ids, "label": labels, "preds_best": predictions})
    for directory in (log_dir, output_dir):
        predictions_df.to_csv(directory / "results.csv", index=False)
        (directory / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    r2 = "undefined" if results["R2"] is None else f"{results['R2']:.5g}"
    print(f"Test MAE={results['test_MAE']:.5g} | R2={r2} | Results: {output_dir / 'results.json'}", flush=True)
    return results
