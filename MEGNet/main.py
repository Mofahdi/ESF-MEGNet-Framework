#!/usr/bin/env python3
"""Train MEGNet with the shared reproducible experiment workflow."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from process.dataset import MDGLDataset, collate_fn_graph
from process.graph_converter import Icoxx2Graph
from train.megnet import MEGNet
from train._training_dgl import ModelLightningModule
from esf_workflow.training import run_training


if __name__ == "__main__":
    run_training(
        "MEGNet", MEGNet, MDGLDataset, Icoxx2Graph,
        ModelLightningModule, collate_fn_graph,
    )
