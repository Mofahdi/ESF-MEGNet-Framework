from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from esf_workflow.splits import generate_cv_splits, load_indices, validate_indices


class SplitTests(unittest.TestCase):
    def test_disjoint_reproducible_folds_and_csv_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "data.csv"
            pd.DataFrame({"id": range(103)}).to_csv(csv, index=False)
            for name in ("first", "second"):
                generate_cv_splits(csv, root / name, seed=42)
            all_test = []
            for fold in range(10):
                first = load_indices(root / "first" / str(fold), fold, csv, 103)
                second = load_indices(root / "second" / str(fold), None, csv, 103)
                for left, right in zip(first, second):
                    np.testing.assert_array_equal(left, right)
                self.assertEqual(len(first[1]), 11)
                all_test.extend(first[2])
            self.assertEqual(sorted(all_test), list(range(103)))
            with self.assertRaises(FileExistsError):
                generate_cv_splits(csv, root / "first")
            pd.DataFrame({"id": list(reversed(range(103)))}).to_csv(csv, index=False)
            with self.assertRaisesRegex(ValueError, "CSV contents/order"):
                load_indices(root / "first/0", 0, csv, 103)

    def test_original_cv_bug_is_rejected_in_every_fold(self):
        indices = np.arange(100)
        for _, test in KFold(10, shuffle=True, random_state=42).split(indices):
            train, val = train_test_split(indices, test_size=0.1, random_state=42)
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_indices(train, val, test, 100)

    def test_invalid_indices(self):
        bad_splits = [
            ([0, 0], [1], [2]), ([0.0], [1], [2]), ([0], [1], [3]),
            ([0], [1], []), ([0], [0], [2]), ([[0]], [1], [2]),
        ]
        for partitions in bad_splits:
            with self.subTest(partitions=partitions), self.assertRaises(ValueError):
                validate_indices(*partitions, 3)
        with self.assertRaisesRegex(ValueError, "cover"):
            validate_indices([0], [1], [2], 4)

    def test_ambiguous_mismatched_and_tampered_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "data.csv"
            pd.DataFrame({"id": range(20)}).to_csv(csv, index=False)
            generate_cv_splits(csv, root / "splits")
            directory = root / "splits/0"
            np.save(directory / "train_idx_1.npy", np.arange(16))
            with self.assertRaisesRegex(ValueError, "specify --fold"):
                load_indices(directory, None, csv, 20)
            load_indices(directory, 0, csv, 20)
            (directory / "train_idx_1.npy").unlink()
            np.save(directory / "train_idx_0.npy", np.arange(16))
            with self.assertRaisesRegex(ValueError, "differs from the split manifest"):
                load_indices(directory, 0, csv, 20)
            (directory / "split_manifest.json").unlink()
            (directory / "val_idx_0.npy").rename(directory / "val_idx_1.npy")
            with self.assertRaisesRegex(ValueError, "different folds"):
                load_indices(directory, None, csv, 20)


if __name__ == "__main__":
    unittest.main()
