"""Exercise the Python workflow and all model entry points on tiny inputs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from fixtures import write_dataset


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("MEGNet", "ESF_MEGNet", "ESF_MEGNet_zero", "ESF_MEGNet_rand")


class WorkflowTests(unittest.TestCase):
    def test_full_workflow_and_repeated_runs(self):
        # Spaces in paths also exercise argument boundaries and CSV-relative material paths.
        with tempfile.TemporaryDirectory(prefix="esf workflow ") as tmp:
            directory = Path(tmp)
            csv = write_dataset(directory)
            output = directory / "experiment"
            # A Zenodo/source archive need not include a .git directory.
            archive = directory / "source archive"
            for folder in (*VARIANTS, "scripts", "esf_workflow"):
                shutil.copytree(ROOT / folder, archive / folder, ignore=shutil.ignore_patterns("__pycache__"))
            env = {
                **os.environ, "N_SPLITS": "2", "MAX_EPOCHS": "2",
                "PATIENCE": "10", "BATCH_SIZE": "4", "DECAY_STEPS": "10", "DEVICE": "cpu",
                "SEED": "42", "FEATURE_SEED": "42", "SPLIT_SEED": "42",
                "MODELS": " ".join(VARIANTS), "DGLBACKEND": "pytorch", "PYTHONHASHSEED": "42",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            }
            env.pop("DATA_ROOT", None)
            env.pop("MATERIAL_ROOT", None)
            def run(command):
                completed = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=300)
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-6000:])
                return completed

            run([sys.executable, str(archive / "scripts/reproduce_target.py"), str(csv), "target", str(output)])
            summary = pd.read_csv(output / "summary.csv")
            self.assertEqual(set(summary.model), set(VARIANTS))
            self.assertTrue((summary.n_folds == 2).all())
            hashes = set()
            for variant in VARIANTS:
                combined = pd.read_csv(output / f"{variant}_out_of_fold.csv")
                np.testing.assert_array_equal(combined.row_index, np.arange(20))
                for fold in range(2):
                    invocation = json.loads((output / variant / f"fold_{fold}/command.json").read_text())
                    self.assertEqual(invocation["argv"][0], sys.executable)
                    self.assertEqual(invocation["cwd"], str(archive))
                    self.assertEqual(invocation["environment"]["PYTHONHASHSEED"], "42")
                    self.assertIn(str(csv), invocation["argv"])
                    self.assertTrue((output / variant / f"fold_{fold}/training.log").stat().st_size > 0)
                    result = json.loads((output / variant / f"fold_{fold}/results.json").read_text())
                    metadata = json.loads(Path(result["run_metadata"]).read_text())
                    self.assertIsNone(metadata["git_commit"])
                    checkpoint = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
                    self.assertEqual(checkpoint["lr_schedulers"][0]["last_epoch"], checkpoint["epoch"] + 1)
                    if variant != "MEGNet":
                        hashes.add(metadata["initial_parameters_sha256"])
                        self.assertEqual(metadata["parameter_count"], 281145)
            # All ESF variants start from exactly the same parameters despite random inputs.
            self.assertEqual(len(hashes), 1)

            repeated = output / "ESF_MEGNet_rand/fold_0"
            original = json.loads((repeated / "results.json").read_text())
            original_predictions = pd.read_csv(repeated / "results.csv")
            command = [sys.executable, str(ROOT / "ESF_MEGNet_rand/main.py"),
                       "--csv-filename", str(csv), "--col-name", "target",
                       "--split-dir", str(output / "splits/0"), "--fold", "0",
                       "--output-dir", str(repeated), "--max-epochs", "2", "--patience", "10",
                       "--batch-size", "4", "--decay-steps", "10", "--device", "cpu"]
            run(command)
            current = json.loads((repeated / "results.json").read_text())
            self.assertNotEqual(current["best_checkpoint"], original["best_checkpoint"])
            self.assertIn("version_1", current["metrics_path"])
            self.assertTrue(Path(original["best_checkpoint"]).exists())
            pd.testing.assert_frame_equal(original_predictions, pd.read_csv(repeated / "results.csv"), check_exact=True)

            # A changed seed gives different weights/results, and the current log is used.
            run(command + ["--seed", "43", "--best-ckpt-name", "best-{epoch:02d}-{val_MAE:.4f}"])
            formatted = json.loads((repeated / "results.json").read_text())
            self.assertIn("epoch=", Path(formatted["best_checkpoint"]).name)
            self.assertIn("version_2", formatted["metrics_path"])
            self.assertNotEqual(formatted["test_MAE"], original["test_MAE"])
            metrics = pd.read_csv(formatted["metrics_path"])
            self.assertAlmostEqual(formatted["val_MAE_min"], metrics.val_MAE.min())


if __name__ == "__main__":
    unittest.main()
