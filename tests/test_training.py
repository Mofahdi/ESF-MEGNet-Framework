from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from esf_workflow.bonds import align_bond_descriptors
from esf_workflow.training import load_best_checkpoint
from fixtures import write_material


VARIANTS = ("MEGNet", "ESF_MEGNet", "ESF_MEGNet_zero", "ESF_MEGNet_rand")


class TrainingTests(unittest.TestCase):
    def test_scheduler_steps_once_per_epoch_in_all_variants(self):
        torch.set_num_threads(1)
        for variant in VARIANTS:
            base = import_module(f"{variant}.train._training_dgl").ModelLightningModule

            class TinyModule(base):
                def step(self, batch):
                    x, y = batch
                    loss = torch.nn.functional.mse_loss(self.model(x), y)
                    return {"Total_Loss": loss, "MAE": loss}, len(x)

            module = TinyModule(torch.nn.Linear(1, 1), decay_steps=10)
            loader = DataLoader(TensorDataset(torch.ones(4, 1), torch.zeros(4, 1)), batch_size=2)
            trainer = pl.Trainer(max_epochs=3, accelerator="cpu", logger=False, enable_checkpointing=False,
                                 enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
            trainer.fit(module, loader, loader)
            self.assertEqual(module.lr_schedulers().last_epoch, 3, variant)
            expected = 0.00001 + (0.001 - 0.00001) * (1 + np.cos(3 * np.pi / 10)) / 2
            self.assertAlmostEqual(module.lr_schedulers().get_last_lr()[0], expected)

    def test_actual_checkpoint_path_and_eval_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(0.5))
            for filename, value in (("best.ckpt", 1.0), ("best-v1.ckpt", 2.0), ("best-epoch=03.ckpt", 3.0)):
                with torch.no_grad():
                    module[0].weight.fill_(value)
                torch.save({"state_dict": module.state_dict()}, root / filename)
            for filename, expected in (("best-v1.ckpt", 2.0), ("best-epoch=03.ckpt", 3.0)):
                module.train()
                actual = load_best_checkpoint(module, SimpleNamespace(best_model_path=str(root / filename)))
                self.assertEqual(Path(actual).name, filename)
                self.assertEqual(module[0].weight.item(), expected)
                self.assertFalse(module.training)
            with self.assertRaisesRegex(RuntimeError, "no best checkpoint"):
                load_best_checkpoint(module, SimpleNamespace(best_model_path=""))

    def test_bond_identity_including_periodic_images(self):
        hp = [([0, 1], 1.0, [0, 0, 0], -1), ([0, 1], 2.0, [1, 0, 0], -2)]
        bi = [([0, 1], 2.0, [1, 0, 0], 0.2), ([0, 1], 1.0, [0, 0, 0], 0.1)]
        self.assertEqual([value for _, value in align_bond_descriptors(hp, bi)], [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            align_bond_descriptors(hp, bi[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            align_bond_descriptors(hp, bi + bi[:1])

    def test_real_file_alignment_and_cache_invalidation(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                material = write_material(root / "material", reverse_icobi=True)
                converter_class = import_module(f"{variant}.process.graph_converter").Icoxx2Graph
                dataset_module = import_module(f"{variant}.process.dataset")
                dataset_class = getattr(dataset_module, "MDGLDataset" if variant == "MEGNet" else "MGLDataset")
                kwargs = dict(
                    converter=converter_class(bond_length_cutoff=5, node_feats_type="Z"),
                    structures_path=[str(material / "CONTCAR.gz")], ids=["sample"],
                    save_dir=str(root / "cache"), name="test_cache",
                )
                if variant == "MEGNet":
                    kwargs["icoxxlists_path"] = [str(material / "ICOHPLIST.lobster.gz")]
                else:
                    kwargs.update(icohplists_path=[str(material / "ICOHPLIST.lobster.gz")],
                                  icobilists_path=[str(material / "ICOBILIST.lobster.gz")],
                                  chgcars_path=[str(material / "CHARGE.lobster.gz")], doscars_path=[None])
                torch.manual_seed(42)
                first = dataset_class(**kwargs, labels={"target": [1.0]}, save_cache=True)
                # Identical settings really load, rather than simply always rebuilding.
                from unittest.mock import patch
                with patch.object(dataset_class, "process", side_effect=AssertionError("cache missed")):
                    loaded = dataset_class(**kwargs, labels={"target": [1.0]}, save_cache=True)
                self.assertEqual(loaded.labels, first.labels)
                fresh = dataset_class(**kwargs, labels={"target": [2.0]}, save_cache=False)
                self.assertEqual(fresh.labels["target"], [2.0])
                changed = dataset_class(**kwargs, labels={"target": [3.0]}, save_cache=True)
                self.assertEqual(changed.labels["target"], [3.0])
                write_material(material, distance=1.2, reverse_icobi=True)
                changed_input = dataset_class(**kwargs, labels={"target": [3.0]}, save_cache=True)
                self.assertFalse(torch.equal(changed.graphs[0].ndata["pos"], changed_input.graphs[0].ndata["pos"]))
                if variant != "MEGNet":
                    graph = changed_input.graphs[0]
                    for src, dst, bi in zip(*[x.tolist() for x in graph.edges()], graph.edata["ICOBIs"].tolist()):
                        self.assertAlmostEqual(bi, 0.1 if {src, dst} == {0, 1} else 0.2)
                    write_material(material, missing_icobi=True)
                    with self.assertRaisesRegex(ValueError, "coverage differs"):
                        dataset_class(**kwargs, labels={"target": [3.0]}, save_cache=False)


if __name__ == "__main__":
    unittest.main()
