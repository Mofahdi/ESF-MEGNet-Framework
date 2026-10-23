#!/usr/bin/env python3
"""Train ESF_MEGNet with the shared reproducible experiment workflow."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from process.dataset import MGLDataset, collate_fn_graph
from process.graph_converter import Icoxx2Graph
from train.esf_megnet import ESF_MEGNet
from train._training_dgl import ModelLightningModule
from esf_workflow.training import run_training


if __name__ == "__main__":
    run_training(
        "ESF_MEGNet", ESF_MEGNet, MGLDataset, Icoxx2Graph,
        ModelLightningModule, collate_fn_graph,
    )
