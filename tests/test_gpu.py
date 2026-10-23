"""Exercise actual CUDA training and checkpoint evaluation for every variant."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from esf_workflow.splits import generate_cv_splits
from fixtures import write_dataset


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("MEGNet", "ESF_MEGNet", "ESF_MEGNet_zero", "ESF_MEGNet_rand")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA PyTorch and an NVIDIA GPU")
class GPUWorkflowTests(unittest.TestCase):
    def test_all_variants_train_on_gpu_by_default(self):
        with tempfile.TemporaryDirectory(prefix="esf cuda ") as tmp:
            directory = Path(tmp)
            csv = write_dataset(directory)
            generate_cv_splits(csv, directory / "splits", seed=42, n_splits=2)
            env = {
                **os.environ, "DGLBACKEND": "pytorch", "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    output = directory / variant
                    command = [
                        sys.executable, str(ROOT / variant / "main.py"),
                        "--csv-filename", str(csv), "--col-name", "target",
                        "--split-dir", str(directory / "splits/0"), "--fold", "0",
                        "--output-dir", str(output), "--max-epochs", "1",
                        "--batch-size", "4",
                    ]
                    completed = subprocess.run(
                        command, cwd=directory, env=env, capture_output=True,
                        text=True, timeout=180,
                    )
                    self.assertEqual(completed.returncode, 0,
                                     completed.stdout[-3000:] + completed.stderr[-6000:])
                    result = json.loads((output / "results.json").read_text())
                    metadata = json.loads(Path(result["run_metadata"]).read_text())
                    self.assertEqual(metadata["arguments"]["device"], "cuda")
                    # Inspect saved tensor locations to verify training really used CUDA.
                    locations = []
                    def record_location(storage, location):
                        locations.append(location)
                        return storage
                    checkpoint = torch.load(result["best_checkpoint"], map_location=record_location,
                                            weights_only=False)
                    self.assertTrue(any(location.startswith("cuda:") for location in locations))
                    self.assertEqual(checkpoint["lr_schedulers"][0]["last_epoch"], 1)
                    predictions = pd.read_csv(output / "results.csv")
                    np.testing.assert_array_equal(predictions.row_index,
                                                  np.load(directory / "splits/0/test_idx_0.npy"))
                    self.assertTrue(np.isfinite(predictions.preds_best).all())


if __name__ == "__main__":
    unittest.main()
