"""Download only the NOMAD raw files used by the configured target tables."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import gzip
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import time
import zipfile
import zlib

import requests

from .splits import file_sha256

API_BASE = "https://nomad-lab.eu/prod/v1/api/v1"
FILES = ("CONTCAR.gz", "ICOHPLIST.lobster.gz", "ICOBILIST.lobster.gz", "CHARGE.lobster.gz")
ROOT = Path(__file__).resolve().parents[1]


def requested_entries(config_path, mapping_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    wanted = set()
    for prop in config["properties"].values():
        with (config_path.parent / prop["csv_filename"]).open(newline="") as handle:
            wanted.update(row["id"] for row in csv.DictReader(handle))
    if not wanted or any(re.fullmatch(r"mp-\d+", mid) is None for mid in wanted):
        raise ValueError("NOMAD downloading requires nonempty Materials Project IDs (mp-<number>)")
    with Path(mapping_path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row["id"]: row["nomad_entry_id"] for row in rows}
    if len(mapping) != len(rows) or any(not entry for entry in mapping.values()):
        raise ValueError("The NOMAD mapping contains duplicate IDs or missing entry IDs")
    missing = wanted - mapping.keys()
    if missing:
        raise ValueError(f"No NOMAD entry mapping for {len(missing)} materials: {', '.join(sorted(missing))}")
    selected = {mid: mapping[mid] for mid in sorted(wanted)}
    if len(set(selected.values())) != len(selected):
        raise ValueError("Different material IDs must not share a NOMAD entry")
    return selected


def is_complete(directory, entry_id):
    """Resume only files whose recorded source and SHA-256 checksums agree."""
    try:
        receipt = json.loads((directory / ".nomad.json").read_text())
        return (receipt["entry_id"] == entry_id and receipt["api_base"] == API_BASE
                and set(receipt["files"]) == set(FILES)
                and all(file_sha256(directory / name) == receipt["files"][name] for name in FILES))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def has_complete_local_inputs(directory):
    """Reuse complete user-supplied materials without claiming NOMAD provenance."""
    if (directory / ".nomad.json").exists():
        # Managed files must match their receipt; valid gzip alone is insufficient.
        return False
    try:
        for name in FILES:
            with gzip.open(directory / name, "rb") as source:
                while source.read(1024 * 1024):
                    pass
    except (OSError, EOFError, zlib.error):
        return False
    return True


def install_archive(archive_path, entries, output_dir):
    """Use NOMAD's manifest to associate each selected entry with its raw folder."""
    by_entry = {entry: mid for mid, entry in entries.items()}
    output_dir = Path(output_dir).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("NOMAD archive contains duplicate file names")
        manifest = json.loads(archive.read("manifest.json"))
        records = {row["entry_id"]: row for row in manifest}
        if len(records) != len(manifest) or set(records) != set(by_entry):
            raise ValueError("NOMAD archive does not contain exactly the requested entries")
        for entry_id, record in records.items():
            mid = by_entry[entry_id]
            folder = PurePosixPath(record["mainfile"]).parent
            if folder.is_absolute() or ".." in folder.parts or not folder.parts:
                raise ValueError(f"Unsafe raw folder in NOMAD manifest: {folder}")
            destination = output_dir / mid
            if destination.resolve().parent != output_dir:
                raise ValueError(f"Material directory escapes output directory: {mid}")
            with tempfile.TemporaryDirectory(prefix="extract-", dir=output_dir / ".downloads") as tmp:
                staging = Path(tmp)
                checksums = {}
                for name in FILES:
                    member = str(folder / name)
                    if member not in names:
                        raise ValueError(f"NOMAD entry {entry_id} ({mid}) is missing {name}")
                    with archive.open(member) as source, (staging / name).open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    # Reading to EOF verifies the gzip CRC, not just its header.
                    with gzip.open(staging / name, "rb") as source:
                        while source.read(1024 * 1024):
                            pass
                    checksums[name] = file_sha256(staging / name)
                receipt = {"api_base": API_BASE, "entry_id": entry_id, "id": mid,
                           "upload_id": record["upload_id"], "mainfile": record["mainfile"],
                           "files": checksums}
                (staging / ".nomad.json").write_text(json.dumps(receipt, indent=2) + "\n")
                destination.mkdir(exist_ok=True)
                for name in FILES:
                    (staging / name).replace(destination / name)
                # Commit the receipt last: interrupted copies cannot be mistaken for a complete entry.
                (staging / ".nomad.json").replace(destination / ".nomad.json")


def fetch_batch(entries, output_dir, attempts=3):
    body = {"owner": "public", "query": {"entry_id:any": list(entries.values())},
            "files": {"include_files": list(FILES)}}
    for attempt in range(attempts):
        try:
            with tempfile.TemporaryDirectory(prefix="batch-", dir=output_dir / ".downloads") as tmp:
                archive_path = Path(tmp) / "raw.zip"
                with requests.post(f"{API_BASE}/entries/raw/query", json=body,
                                   stream=True, timeout=(30, 300)) as response:
                    response.raise_for_status()
                    with archive_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            handle.write(chunk)
                install_archive(archive_path, entries, output_dir)
            return
        except (requests.RequestException, OSError, ValueError, KeyError, zipfile.BadZipFile):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def download_for_config(config_path, output_dir=None, mapping_path=None, batch_size=50, workers=2):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if output_dir is None:
        base = config.get("material_root")
        if not isinstance(base, str) or not base.strip():
            raise ValueError("Set material_root in the configuration or pass --output-dir")
        output_dir = config_path.parent / base
    output_dir = Path(output_dir).resolve()
    mapping_path = Path(mapping_path or ROOT / "data/nomad_entry_ids.csv").resolve()
    if batch_size < 1 or workers < 1:
        raise ValueError("batch_size and workers must be positive")
    entries = requested_entries(config_path, mapping_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".downloads").mkdir(exist_ok=True)
    local_materials = []
    pending = {}
    for mid, entry in entries.items():
        directory = output_dir / mid
        if is_complete(directory, entry):
            continue
        if has_complete_local_inputs(directory):
            local_materials.append(mid)
        else:
            pending[mid] = entry
    completed = len(entries) - len(pending)
    print(f"NOMAD: {len(entries)} materials requested; {completed} complete locally; {len(pending)} to download.", flush=True)
    state = {"api_base": API_BASE, "config_sha256": file_sha256(config_path),
             "mapping_sha256": file_sha256(mapping_path), "requested": len(entries),
             "complete": completed, "local_materials": local_materials,
             "failures": {}, "status": "downloading"}

    def save_state():
        temporary = output_dir / "download_manifest.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output_dir / "download_manifest.json")

    save_state()
    items = list(pending.items())
    batches = [dict(items[start:start + batch_size]) for start in range(0, len(items), batch_size)]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_batch, batch, output_dir): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                future.result()
                state["complete"] += len(batch)
            except Exception as error:
                # Some entries may already have been installed before another failed.
                for mid, entry in batch.items():
                    if is_complete(output_dir / mid, entry):
                        state["complete"] += 1
                    else:
                        state["failures"][mid] = str(error)
            save_state()
            print(f"NOMAD: {state['complete']}/{len(entries)} complete; {len(state['failures'])} failed.", flush=True)
    state["status"] = "failed" if state["failures"] else "complete"
    save_state()
    if state["failures"]:
        raise RuntimeError(f"Download incomplete; see {output_dir / 'download_manifest.json'}. Rerun to retry; no samples were dropped.")
    return state
