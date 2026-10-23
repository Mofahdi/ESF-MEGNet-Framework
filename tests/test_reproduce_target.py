"""Check that child training failures preserve logs and stop the Python runner."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.reproduce_target import run_target


class PythonRunnerTests(unittest.TestCase):
    def test_child_failure_preserves_both_output_streams_and_stops_remaining_folds(self):
        with tempfile.TemporaryDirectory(prefix="esf runner ") as tmp:
            root = Path(tmp)
            csv = root / "targets.csv"
            csv.write_text("id,mat_dir,target\n" + "".join(f"sample_{i},unused,{i}\n" for i in range(20)))
            model = root / "MEGNet"
            model.mkdir()
            (model / "main.py").write_text(
                "import os, sys\n"
                "print('hash seed=' + os.environ['PYTHONHASHSEED'], flush=True)\n"
                "print('intentional training failure', file=sys.stderr, flush=True)\n"
                "sys.exit(17)\n"
            )
            output = root / "results"
            captured = io.StringIO()
            with patch("scripts.reproduce_target.ROOT", root), contextlib.redirect_stdout(captured):
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    run_target(csv, "target", output, settings={"seed": 37, "n_splits": 2, "models": ["MEGNet"]})
            self.assertEqual(error.exception.returncode, 17)
            log = (output / "MEGNet/fold_0/training.log").read_text()
            for message in ("hash seed=37", "intentional training failure"):
                self.assertIn(message, log)
                self.assertIn(message, captured.getvalue())
            invocation = json.loads((output / "MEGNet/fold_0/command.json").read_text())
            self.assertEqual(invocation["argv"][0], sys.executable)
            self.assertEqual(invocation["environment"]["PYTHONHASHSEED"], "37")
            self.assertIn("torch==", (output / "environment.txt").read_text())
            self.assertIn("Git metadata unavailable", (output / "git_commit.txt").read_text())
            self.assertFalse((output / "MEGNet/fold_1").exists())
            self.assertFalse((output / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
