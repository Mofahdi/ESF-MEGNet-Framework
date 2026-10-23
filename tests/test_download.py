"""Check material identity, interrupted downloads, and archive integrity offline."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from esf_workflow.download import FILES, download_for_config, install_archive, is_complete, requested_entries


def make_archive(path, entries, missing=None, wrong_entry=False, corrupt=False):
    manifest = []
    with zipfile.ZipFile(path, "w") as archive:
        for mid, entry in entries.items():
            folder = f"upload/{mid}"
            manifest.append({"entry_id": "unexpected" if wrong_entry else entry,
                             "upload_id": "upload", "mainfile": f"{folder}/lobsterout.gz"})
            for filename in FILES:
                if filename != missing:
                    payload = gzip.compress(f"{mid}: {filename}\n".encode())
                    archive.writestr(f"{folder}/{filename}", b"invalid gzip" if corrupt else payload)
        # An unrelated member must never be extracted.
        archive.writestr("../escaped.txt", "not a requested file")
        archive.writestr("manifest.json", json.dumps(manifest))


class DownloadTests(unittest.TestCase):
    def test_preserves_complete_local_materials_when_others_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "targets.csv").write_text("id,target\nmp-1,1.0\nmp-2,2.0\n")
            config = root / "properties.json"
            config.write_text(json.dumps({"material_root": "materials", "properties": {
                "target": {"csv_filename": "targets.csv"}}}))
            mapping = root / "mapping.csv"
            mapping.write_text("id,nomad_entry_id\nmp-1,entry1\nmp-2,entry2\n")
            local = root / "materials/mp-1"
            local.mkdir(parents=True)
            originals = {name: gzip.compress(f"Local input: {name}".encode()) for name in FILES}
            for name, contents in originals.items():
                (local / name).write_bytes(contents)

            def fetch(entries, output):
                archive = root / "raw.zip"
                make_archive(archive, entries)
                install_archive(archive, entries, output)

            with patch("esf_workflow.download.fetch_batch", side_effect=fetch) as download:
                state = download_for_config(config, mapping_path=mapping, workers=1)
                download.assert_called_once_with({"mp-2": "entry2"}, root / "materials")
                self.assertEqual(state["complete"], 2)
                self.assertEqual(state["local_materials"], ["mp-1"])
                self.assertFalse((local / ".nomad.json").exists())
                self.assertEqual({name: (local / name).read_bytes() for name in FILES}, originals)
                download_for_config(config, mapping_path=mapping, workers=1)
                self.assertEqual(download.call_count, 1)
                (local / FILES[0]).write_bytes(originals[FILES[0]][:-4])
                state = download_for_config(config, mapping_path=mapping, workers=1)
                self.assertEqual(download.call_args.args[0], {"mp-1": "entry1"})
                self.assertEqual(state["local_materials"], [])
                self.assertTrue(is_complete(local, "entry1"))

    def test_manifest_identity_file_integrity_and_safe_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "materials"
            (output / ".downloads").mkdir(parents=True)
            entries = {"mp-1": "entry1", "mp-2": "entry2"}
            archive = root / "raw.zip"
            make_archive(archive, entries)
            install_archive(archive, entries, output)
            for mid, entry in entries.items():
                self.assertTrue(is_complete(output / mid, entry))
                with gzip.open(output / mid / FILES[0], "rt") as handle:
                    self.assertTrue(handle.read().startswith(mid + ":"))
            self.assertFalse((root / "escaped.txt").exists())
            self.assertFalse(is_complete(output / "mp-1", "different-entry"))
            (output / "mp-1" / FILES[0]).write_bytes(b"interrupted")
            self.assertFalse(is_complete(output / "mp-1", "entry1"))
            for options in ({"wrong_entry": True}, {"missing": FILES[-1]}, {"corrupt": True}):
                fresh = root / f"bad-{len(list(root.iterdir()))}"
                (fresh / ".downloads").mkdir(parents=True)
                make_archive(archive, entries, **options)
                with self.assertRaises((ValueError, OSError)):
                    install_archive(archive, entries, fresh)
                self.assertFalse(is_complete(fresh / "mp-1", "entry1"))

    def test_resumes_only_verified_files_and_reports_missing_mappings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "targets.csv").write_text("id,target\nmp-1,1.0\nmp-2,2.0\n")
            config = root / "properties.json"
            config.write_text(json.dumps({"material_root": "materials", "properties": {
                "first": {"csv_filename": "targets.csv"}, "second": {"csv_filename": "targets.csv"}}}))
            mapping = root / "mapping.csv"
            mapping.write_text("id,nomad_entry_id\nmp-1,entry1\nmp-2,entry2\nmp-3,unneeded\n")
            self.assertEqual(requested_entries(config, mapping), {"mp-1": "entry1", "mp-2": "entry2"})
            batches = []

            def fetch(entries, output):
                batches.append(entries.copy())
                archive = root / "raw.zip"
                make_archive(archive, entries)
                install_archive(archive, entries, output)

            with patch("esf_workflow.download.fetch_batch", side_effect=fetch):
                state = download_for_config(config, mapping_path=mapping, workers=1)
                self.assertEqual(state["complete"], 2)
                self.assertEqual(len(batches), 1)
                download_for_config(config, mapping_path=mapping, workers=1)
                self.assertEqual(len(batches), 1)
                (root / "materials/mp-2" / FILES[1]).unlink()
                download_for_config(config, mapping_path=mapping, workers=1)
                self.assertEqual(batches[-1], {"mp-2": "entry2"})
            (root / "materials/mp-2" / FILES[1]).unlink()
            with patch("esf_workflow.download.fetch_batch", side_effect=RuntimeError("network unavailable")):
                with self.assertRaisesRegex(RuntimeError, "no samples were dropped"):
                    download_for_config(config, mapping_path=mapping)
            state = json.loads((root / "materials/download_manifest.json").read_text())
            self.assertEqual(state["complete"], 1)
            self.assertEqual(set(state["failures"]), {"mp-2"})
            mapping.write_text("id,nomad_entry_id\nmp-1,entry1\n")
            with self.assertRaisesRegex(ValueError, "mp-2"):
                requested_entries(config, mapping)


if __name__ == "__main__":
    unittest.main()
