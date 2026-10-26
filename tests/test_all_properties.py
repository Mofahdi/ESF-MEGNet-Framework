"""Validate the five-property launcher and run it on distinct synthetic targets."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fixtures import write_dataset
from scripts.reproduce import PROPERTY_TARGETS, main, read_plan, run_all


ROOT = Path(__file__).resolve().parents[1]


def make_config(directory):
    csv = write_dataset(directory / "data")
    frame = pd.read_csv(csv)
    columns = ("frequency", "log_msd", "log_k", "log_g", "log_hv")
    for index, column in enumerate(columns):
        # Negative logged targets are valid, and each property has distinct labels.
        frame[column] = frame.target * (index + 1) - 2 * index
    frame.to_csv(csv, index=False)
    config_dir = directory / "configuration"
    config_dir.mkdir()
    entries = {
        name: {"csv_filename": "../data/dataset.csv", "col_name": column, "target_scale": scale}
        for (name, (scale, _)), column in zip(PROPERTY_TARGETS.items(), columns)
    }
    config = {"properties": entries}
    path = config_dir / "properties.json"
    path.write_text(json.dumps(config))
    return path, config, frame


class AllPropertyTests(unittest.TestCase):
    def test_check_only_resolves_inputs_without_training_or_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, config, frame = make_config(directory)
            # A per-property data root is relative to the config, not the calling directory.
            config["properties"]["bulk_modulus"]["data_root"] = "../data"
            path.write_text(json.dumps(config))
            destination = directory / "unused output"
            with patch("scripts.reproduce.run_target") as run, \
                    patch("esf_workflow.download.download_for_config") as download, \
                    contextlib.redirect_stdout(io.StringIO()):
                plan = run_all(path, destination, check_only=True)
            run.assert_not_called()
            download.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertEqual(plan["total_fits"], 200)
            self.assertEqual(len(plan["properties"]), 5)
            self.assertTrue(all(entry["n_samples"] == len(frame) for entry in plan["properties"].values()))
            self.assertEqual(plan["properties"]["average_msd"]["target_scale"], "log10")
            self.assertEqual(plan["properties"]["bulk_modulus"]["data_root"], str(directory / "data"))

    def test_last_properties_checked_before_first_training_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, config, _ = make_config(directory)
            config["properties"]["shear_modulus"]["col_name"] = "missing_column"
            config["properties"]["vickers_hardness"]["csv_filename"] = "missing_file.csv"
            path.write_text(json.dumps(config))
            with patch("scripts.reproduce.run_target") as run:
                with self.assertRaises(ValueError) as error:
                    run_all(path, directory / "results")
            self.assertIn("shear_modulus", str(error.exception))
            self.assertIn("vickers_hardness", str(error.exception))
            run.assert_not_called()
            self.assertFalse((directory / "results").exists())

    def test_material_root_overrides_old_absolute_paths_without_changing_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, config, frame = make_config(directory)
            frame["mat_dir"] = [f"/unavailable/hpc/{mid}" for mid in frame.id]
            csv = directory / "data/dataset.csv"
            frame.to_csv(csv, index=False)
            original = csv.read_bytes()
            config["material_root"] = "../data/materials"
            path.write_text(json.dumps(config))
            plan = read_plan(path)
            self.assertTrue(all(entry["material_root"] == str(directory / "data/materials")
                                for entry in plan["properties"].values()))
            self.assertEqual(csv.read_bytes(), original)
            frame.loc[0, "id"] = "../outside"
            frame.to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, "single directory name"):
                read_plan(path)

    def test_missing_inputs_are_downloaded_automatically_before_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, config, _ = make_config(directory)
            config["material_root"] = "../data/materials"
            path.write_text(json.dumps(config))
            material_file = directory / "data/materials/sample_0/CONTCAR.gz"
            original = material_file.read_bytes()
            material_file.unlink()
            def prepare(config_path):
                self.assertEqual(config_path, path)
                material_file.write_bytes(original)
            with patch("esf_workflow.download.download_for_config", side_effect=prepare) as download, \
                    patch("scripts.reproduce.run_target") as train, \
                    contextlib.redirect_stdout(io.StringIO()):
                plan = run_all(path, directory / "unused", check_only=True)
            download.assert_called_once_with(path)
            train.assert_not_called()
            self.assertEqual(plan["total_fits"], 200)
            self.assertFalse((directory / "unused").exists())
            material_file.unlink()
            # Do not proceed if the downloader returns without providing every input.
            with patch("esf_workflow.download.download_for_config"), \
                    patch("scripts.reproduce.run_target") as train, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "required input files are missing"):
                    run_all(path, directory / "unused")
            train.assert_not_called()
            self.assertFalse((directory / "unused").exists())
            config["properties"]["bulk_modulus"]["col_name"] = "unknown"
            path.write_text(json.dumps(config))
            with patch("esf_workflow.download.download_for_config") as download:
                with self.assertRaisesRegex(ValueError, "missing CSV columns"):
                    run_all(path, directory / "unused", check_only=True)
            download.assert_not_called()

    def test_no_argument_entry_point_reuses_matching_runs_and_preserves_other_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, _, _ = make_config(directory)
            path.parent.rename(directory / "configs")
            original = directory / "runs/all_properties_seed42"
            original.mkdir(parents=True)
            sentinel = original / "previous_results.csv"
            sentinel.write_text("preserve these results\n")
            collision = directory / "runs/all_properties_seed42_2"
            collision.write_text("preserve this file too\n")
            with patch("scripts.reproduce.ROOT", directory), \
                    patch.object(sys, "argv", ["reproduce.py"]), \
                    patch("scripts.reproduce.run_target") as train, \
                    patch("scripts.reproduce.combine_summaries", return_value=directory / "summary.csv"), \
                    patch("esf_workflow.download.download_for_config") as download, \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
                main()
            self.assertEqual(train.call_count, 10)
            download.assert_not_called()
            self.assertEqual(sentinel.read_text(), "preserve these results\n")
            self.assertEqual(collision.read_text(), "preserve this file too\n")
            result_dir = directory / "runs/all_properties_seed42_3"
            state = json.loads((result_dir / "run_all.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["settings"]["seed"], 42)
            self.assertEqual(state["total_fits"], 200)
            self.assertEqual(state["output_dir"], str(result_dir))
            self.assertFalse((directory / "runs/all_properties_seed42_4").exists())

    def test_managed_downloads_are_verified_even_when_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, config, _ = make_config(directory)
            config["material_root"] = "../data/materials"
            path.write_text(json.dumps(config))
            (directory / "data/materials/download_manifest.json").write_text("{}")
            with patch("esf_workflow.download.download_for_config", side_effect=RuntimeError("Repair failed")) as download, \
                    patch("scripts.reproduce.run_target") as train, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Repair failed"):
                    run_all(path, directory / "unused")
            download.assert_called_once_with(path)
            train.assert_not_called()
            self.assertFalse((directory / "unused").exists())

    def test_changed_settings_create_a_separate_run_and_explicit_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, config, _ = make_config(directory)
            with patch("scripts.reproduce.ROOT", directory), \
                    patch("scripts.reproduce.run_target"), \
                    patch("scripts.reproduce.combine_summaries", return_value=directory / "summary.csv"), \
                    contextlib.redirect_stdout(io.StringIO()):
                first = run_all(path)
                original = (Path(first["output_dir"]) / "run_all.json").read_bytes()
                config["settings"] = {"max_epochs": 2}
                path.write_text(json.dumps(config))
                second = run_all(path)
                self.assertNotEqual(first["output_dir"], second["output_dir"])
                self.assertEqual((Path(first["output_dir"]) / "run_all.json").read_bytes(), original)
                with self.assertRaisesRegex(ValueError, "different data/settings"):
                    run_all(path, first["output_dir"])

    def test_invalid_targets_scales_settings_and_missing_material_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, config, frame = make_config(directory)
            config["settings"] = {"models": ["MEGNet", "MEGNet"]}
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_plan(path)
            config.pop("settings")
            config["properties"]["average_msd"]["target_scale"] = "linear"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "already use that scale"):
                read_plan(path)
            config["properties"]["average_msd"]["target_scale"] = "log10"
            path.write_text(json.dumps(config))
            frame.loc[0, "log_hv"] = np.inf
            frame.to_csv(directory / "data/dataset.csv", index=False)
            with self.assertRaisesRegex(ValueError, "vickers_hardness: target values must be finite"):
                read_plan(path)
            frame.loc[0, "log_hv"] = 1.0
            frame.to_csv(directory / "data/dataset.csv", index=False)
            (directory / "data/materials/sample_0/CHARGE.lobster.gz").unlink()
            with self.assertRaisesRegex(ValueError, "required input files are missing"):
                read_plan(path)

    def test_failure_is_recorded_and_later_properties_do_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path, _, _ = make_config(directory)
            destination = directory / "results"
            failure = subprocess.CalledProcessError(17, [sys.executable, "main.py"])
            with patch("scripts.reproduce.run_target", side_effect=[None, failure]) as run:
                with self.assertRaises(subprocess.CalledProcessError), contextlib.redirect_stdout(io.StringIO()):
                    run_all(path, destination)
            self.assertEqual(run.call_count, 2)
            state = json.loads((destination / "run_all.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["property_status"]["phonon_dos_peak"], "complete")
            self.assertEqual(state["property_status"]["average_msd"], "failed")
            self.assertEqual(state["property_status"]["bulk_modulus"], "pending")
            self.assertFalse((destination / "summary_all_properties.csv").exists())
            with patch("scripts.reproduce.run_target") as resumed, \
                    patch("scripts.reproduce.combine_summaries", return_value=destination / "summary.csv"), \
                    contextlib.redirect_stdout(io.StringIO()):
                run_all(path, destination)
            self.assertEqual(resumed.call_count, 5)
            self.assertEqual(json.loads((destination / "run_all.json").read_text())["status"], "complete")

    def test_all_five_targets_through_real_training_workflow(self):
        with tempfile.TemporaryDirectory(prefix="esf five properties ") as tmp:
            directory = Path(tmp)
            path, config, frame = make_config(directory)
            config["settings"] = {
                "seed": 37, "feature_seed": 11, "split_seed": 7,
                "n_splits": 2, "max_epochs": 1, "batch_size": 4,
                "models": ["ESF_MEGNet_zero"], "device": "cpu",
            }
            # Exercise the complete runner using IDs to override obsolete absolute paths.
            config["material_root"] = "../data/materials"
            frame["mat_dir"] = [f"/unavailable/hpc/{mid}" for mid in frame.id]
            frame.to_csv(directory / "data/dataset.csv", index=False)
            path.write_text(json.dumps(config))
            # Run the actual no-argument command from outside a portable source copy.
            archive = directory / "source archive"
            for folder in ("ESF_MEGNet_zero", "scripts", "esf_workflow"):
                shutil.copytree(ROOT / folder, archive / folder, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(directory / "data", archive / "data")
            shutil.copytree(path.parent, archive / "configs")
            destination = archive / "runs/all_properties_seed37"
            env = {
                **os.environ, "DGLBACKEND": "pytorch", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                # Explicit JSON settings must override these inherited values.
                "SEED": "999", "MODELS": "MEGNet", "DEVICE": "cuda", "DATA_ROOT": "/not/the/data",
                "MATERIAL_ROOT": "/not/the/materials",
            }
            completed = subprocess.run(
                [sys.executable, str(archive / "scripts/reproduce.py")],
                cwd=directory, env=env, capture_output=True, text=True, timeout=240,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-6000:])
            self.assertIn("Epoch 1/1 | train MAE=", completed.stdout)
            self.assertNotIn("it/s", completed.stdout + completed.stderr)
            state = json.loads((destination / "run_all.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(set(state["property_status"].values()), {"complete"})
            self.assertEqual(state["total_fits"], 10)
            summary = pd.read_csv(destination / "summary_all_properties.csv")
            self.assertEqual(set(summary.property), set(PROPERTY_TARGETS))
            self.assertEqual(len(summary), 5)
            self.assertTrue((summary.model == "ESF_MEGNet_zero").all())
            reference_splits = {}
            for name, entry in config["properties"].items():
                predictions = pd.read_csv(destination / name / "ESF_MEGNet_zero_out_of_fold.csv")
                np.testing.assert_allclose(predictions.label, frame[entry["col_name"]])
                for fold in range(2):
                    indices = np.load(destination / name / "splits" / str(fold) / f"test_idx_{fold}.npy")
                    if fold in reference_splits:
                        np.testing.assert_array_equal(indices, reference_splits[fold])
                    else:
                        reference_splits[fold] = indices
                    result_path = destination / name / "ESF_MEGNet_zero" / f"fold_{fold}" / "results.json"
                    result = json.loads(result_path.read_text())
                    metadata = json.loads(Path(result["run_metadata"]).read_text())
                    self.assertEqual(metadata["arguments"]["col_name"], entry["col_name"])
                    self.assertEqual(result["seed"], 37)
                    self.assertEqual(result["feature_seed"], 11)
                    self.assertEqual(metadata["arguments"]["device"], "cpu")

            # Rerun the actual entry point: only the fold with missing results is fitted.
            finished = list(destination.glob("*/*/fold_*/results.json"))
            preserved = {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in finished[1:]}
            finished[0].unlink()
            resumed = subprocess.run(
                [sys.executable, str(archive / "scripts/reproduce.py")],
                cwd=directory, env=env, capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout[-3000:] + resumed.stderr[-6000:])
            self.assertEqual(resumed.stdout.count("Skipping completed"), 9)
            self.assertEqual(resumed.stdout.count("Epoch 1/1"), 1)
            self.assertTrue(finished[0].is_file())
            for file, original in preserved.items():
                self.assertEqual((file.read_bytes(), file.stat().st_mtime_ns), original)
            self.assertFalse(destination.with_name(destination.name + "_2").exists())


if __name__ == "__main__":
    unittest.main()
