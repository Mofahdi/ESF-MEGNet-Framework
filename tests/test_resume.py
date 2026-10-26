"""Exercise interruption recovery without loading models or rerunning completed folds."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from esf_workflow.resume import run_lock
from esf_workflow.splits import file_sha256
from scripts.reproduce_target import run_target


def finish_fold(command, log_path, env, lock_fd=None):
    """Emulate a finished child using its actual invocation and saved split files."""
    flags = dict(zip(command[3::2], command[4::2]))
    directory = Path(flags["--output-dir"])
    # A restarted fit must start in an empty fold directory, without old weights.
    assert not (directory / "partial.ckpt").exists()
    log_path.write_text("finished training\n")
    frame = pd.read_csv(flags["--csv-filename"])
    fold = int(flags["--fold"])
    log = directory / "logs/MEGNet_training/version_0"
    (log / "splits").mkdir(parents=True)
    for name in ("train", "val", "test"):
        source = Path(flags["--split-dir"]) / f"{name}_idx_{fold}.npy"
        shutil.copyfile(source, log / "splits" / f"{name}_idx.npy")
    indices = np.load(log / "splits/test_idx.npy")
    arguments = {key.removeprefix("--").replace("-", "_"): value for key, value in flags.items()}
    for key in ("seed", "feature_seed", "fold", "max_epochs", "patience", "batch_size", "decay_steps"):
        arguments[key] = int(arguments[key])
    checksum = file_sha256(flags["--csv-filename"])
    (log / "run.json").write_text(json.dumps({"arguments": arguments, "csv_sha256": checksum}))
    (log / "metrics.csv").write_text("epoch,val_MAE\n0,0.5\n")
    labels = frame.iloc[indices][flags["--col-name"]].to_numpy()
    pd.DataFrame({"row_index": indices, "id": frame.iloc[indices].id,
                  "label": labels, "preds_best": labels + 0.5}).to_csv(directory / "results.csv", index=False)
    (directory / "results.json").write_text(json.dumps({
        "variant": directory.parent.name, "fold": fold, "csv_sha256": checksum,
        "seed": arguments["seed"], "feature_seed": arguments["feature_seed"],
        "train_MAE_min": 0.5, "val_MAE_min": 0.5, "test_MAE": 0.5, "R2": 0.5,
        "run_metadata": str(log / "run.json"), "metrics_path": str(log / "metrics.csv"),
    }))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="esf resume ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.csv = self.root / "targets.csv"
        pd.DataFrame({"id": [f"sample_{i}" for i in range(20)], "mat_dir": "unused",
                      "target": np.arange(20, dtype=float)}).to_csv(self.csv, index=False)
        self.output = self.root / "results"
        self.settings = {"n_splits": 2, "models": ["MEGNet", "ESF_MEGNet_zero"], "device": "cpu"}

    def run_target(self, settings=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            run_target(self.csv, "target", self.output, settings=settings or self.settings, **kwargs)
        return output.getvalue()

    def use_absolute_material_paths(self):
        material_root = self.root / "materials"
        frame = pd.read_csv(self.csv)
        frame["mat_dir"] = [str(material_root / mid) for mid in frame.id]
        frame.to_csv(self.csv, index=False)
        return material_root

    def test_legacy_material_root_override_reuses_identical_inputs(self):
        material_root = self.use_absolute_material_paths()
        with patch("scripts.reproduce_target.run_logged", side_effect=finish_fold):
            self.run_target()
        # The old launcher ignored --material-root and wrote no target manifest.
        (self.output / "target_run.json").unlink()
        finished = self.output / "MEGNet/fold_0/results.json"
        old_result = finished.read_bytes(), finished.stat().st_mtime_ns
        command = finished.parent / "command.json"
        old_command = command.read_bytes()
        (self.output / "ESF_MEGNet_zero/fold_1/results.json").unlink()
        with patch("scripts.reproduce_target.run_logged", side_effect=finish_fold) as child:
            self.run_target(material_root=material_root)
        self.assertEqual(child.call_count, 1)
        self.assertEqual((finished.read_bytes(), finished.stat().st_mtime_ns), old_result)
        self.assertEqual(command.read_bytes(), old_command)
        # Both forms remain equivalent after a new target manifest has been saved.
        with patch("scripts.reproduce_target.run_logged") as child:
            self.run_target()
            self.run_target(material_root=material_root)
        child.assert_not_called()

    def test_equivalent_argument_order_and_default_data_root_are_accepted(self):
        self.use_absolute_material_paths()
        with patch("scripts.reproduce_target.run_logged", side_effect=finish_fold):
            self.run_target(data_root=self.csv.parent)
        command = self.output / "MEGNet/fold_0/command.json"
        saved = json.loads(command.read_text())
        pairs = list(zip(saved["argv"][3::2], saved["argv"][4::2]))
        saved["argv"] = saved["argv"][:3] + [value for pair in reversed(pairs) for value in pair]
        command.write_text(json.dumps(saved))
        with patch("scripts.reproduce_target.run_logged") as child:
            self.run_target()
            # This root is unused because every material path is absolute.
            self.run_target(data_root=self.root / "different CSV base")
        child.assert_not_called()

    def test_changed_material_paths_and_saved_training_arguments_are_rejected(self):
        self.use_absolute_material_paths()
        with patch("scripts.reproduce_target.run_logged", side_effect=finish_fold):
            self.run_target()
        with patch("scripts.reproduce_target.run_logged") as child:
            with self.assertRaisesRegex(ValueError, "Material paths differ for 20 rows; sample_0"):
                self.run_target(material_root=self.root / "other materials")
            (self.output / "target_run.json").unlink()
            with self.assertRaisesRegex(ValueError, "Material paths differ for 20 rows; sample_0"):
                self.run_target(material_root=self.root / "other materials")
            command = self.output / "MEGNet/fold_0/command.json"
            original = command.read_text()
            for flag, changed in (("--seed", "99"), ("--col-name", "other_target"), ("--decay-steps", "500")):
                with self.subTest(flag=flag):
                    saved = json.loads(original)
                    saved["argv"][saved["argv"].index(flag) + 1] = changed
                    command.write_text(json.dumps(saved))
                    with self.assertRaisesRegex(ValueError, f"{flag}: saved='{changed}', requested="):
                        self.run_target()
            command.write_text(original)
        child.assert_not_called()

    def test_resume_skips_complete_folds_and_preserves_interrupted_output(self):
        def interrupted(command, log, env, lock_fd=None):
            if log.parent.name == "fold_1":
                (log.parent / "partial.ckpt").write_text("incomplete weights")
                log.write_text("interrupted training\n")
                raise subprocess.CalledProcessError(17, command)
            finish_fold(command, log, env, lock_fd)

        with patch("scripts.reproduce_target.run_logged", side_effect=interrupted) as child:
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_target()
        self.assertEqual(child.call_count, 2)
        finished = self.output / "MEGNet/fold_0/results.json"
        original = finished.read_bytes(), finished.stat().st_mtime_ns
        with patch("scripts.reproduce_target.run_logged", side_effect=finish_fold) as child:
            output = self.run_target()
        self.assertEqual(child.call_count, 3)
        self.assertIn("Skipping completed MEGNet, fold 1/2", output)
        self.assertEqual((finished.read_bytes(), finished.stat().st_mtime_ns), original)
        self.assertEqual((self.output / "_incomplete/MEGNet_fold_1_1/partial.ckpt").read_text(), "incomplete weights")
        self.assertTrue((self.output / "summary.csv").is_file())
        with patch("scripts.reproduce_target.run_logged") as child:
            self.run_target()
        child.assert_not_called()

    def test_legacy_runs_are_reused_and_missing_or_partial_results_are_restarted(self):
        with patch("scripts.reproduce_target.run_logged", side_effect=finish_fold):
            self.run_target()
        (self.output / "target_run.json").unlink()
        with patch("scripts.reproduce_target.run_logged") as child:
            self.run_target()
        child.assert_not_called()
        # Without a target manifest, a missing invocation cannot establish which
        # material directory supplied a legacy fold: fit it again rather than trust it.
        (self.output / "target_run.json").unlink()
        (self.output / "MEGNet/fold_0/command.json").unlink()
        with patch("scripts.reproduce_target.run_logged", side_effect=finish_fold) as child:
            self.run_target()
        self.assertEqual(child.call_count, 1)
        (self.output / "MEGNet/fold_0/results.csv").unlink()
        (self.output / "ESF_MEGNet_zero/fold_1/results.json").write_text('{"unfinished":')
        with patch("scripts.reproduce_target.run_logged", side_effect=finish_fold) as child:
            self.run_target()
        self.assertEqual(child.call_count, 2)
        self.assertTrue((self.output / "_incomplete/ESF_MEGNet_zero_fold_1_1/results.json").exists())

    def test_changed_data_settings_or_splits_are_not_mixed_with_existing_results(self):
        with patch("scripts.reproduce_target.run_logged", side_effect=finish_fold):
            self.run_target()
        with patch("scripts.reproduce_target.run_logged") as child:
            with self.assertRaisesRegex(ValueError, "different data/settings"):
                self.run_target({**self.settings, "seed": 123})
            original = self.csv.read_bytes()
            self.csv.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "different data/settings"):
                self.run_target()
            self.csv.write_bytes(original)
            index_file = self.output / "splits/0/train_idx_0.npy"
            indices = np.load(index_file)
            np.save(index_file, indices[::-1])
            with self.assertRaisesRegex(ValueError, "differs from the split manifest"):
                self.run_target()
        child.assert_not_called()

    def test_child_keeps_run_locked_after_parent_closes_its_descriptor(self):
        child = None
        try:
            with run_lock(self.output) as descriptor:
                child = subprocess.Popen(
                    [sys.executable, "-c", "import sys; sys.stdin.read()"],
                    stdin=subprocess.PIPE, pass_fds=(descriptor,),
                )
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                with run_lock(self.output):
                    pass
            child.communicate(timeout=10)
            with run_lock(self.output):
                pass
        finally:
            if child is not None and child.poll() is None:
                child.kill()
                child.communicate()


if __name__ == "__main__":
    unittest.main()
