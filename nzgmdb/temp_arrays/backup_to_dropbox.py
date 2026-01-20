#!/usr/bin/env python3

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict

import typer

app = typer.Typer(pretty_exceptions_enable=False)

MANIFEST_HEADER = [
    "type",
    "net",
    "name",
    "local_path",
    "zip_name",
    "status",
    "bytes",
]


# ----------------------------
# RCLONE / ZIP HELPERS
# ----------------------------


def zip_directory(src_dir: Path, out_dir: Path) -> Path:
    """
    Zip an entire directory using system zip (fast + reliable).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{src_dir.name}.zip"

    if zip_path.exists():
        return zip_path

    subprocess.check_call(
        ["zip", "-r", "-1", str(zip_path), src_dir.name],
        cwd=src_dir.parent,
    )

    return zip_path


def upload_and_verify(local_file: Path, dropbox_dir: str) -> bool:
    """
    Upload using rclone and verify by file size.
    """
    subprocess.check_call(
        ["rclone", "copy", str(local_file), dropbox_dir],
    )

    local_size = local_file.stat().st_size

    out = (
        subprocess.check_output(
            [
                "rclone",
                "lsf",
                "--format=s",
                f"{dropbox_dir}/{local_file.name}",
            ]
        )
        .decode()
        .strip()
    )

    return bool(out) and int(out) == local_size


# ----------------------------
# MANIFEST LOGIC
# ----------------------------


def load_manifest(path: Path) -> Dict[str, dict]:
    """
    Load manifest into dict keyed by zip_name.
    """
    rows = {}

    if not path.exists():
        return rows

    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["zip_name"]] = row

    return rows


def append_manifest_row(path: Path, row: dict):
    new_file = not path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def update_manifest_status(path: Path, zip_name: str, status: str, size: int):
    rows = load_manifest(path)
    rows[zip_name]["status"] = status
    rows[zip_name]["bytes"] = size

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(rows.values())


# ----------------------------
# DISCOVERY
# ----------------------------


def discover_stationxml(stationxml_dir: Path):
    yield {
        "type": "stationxml",
        "net": "",
        "name": stationxml_dir.name,
        "local_path": str(stationxml_dir),
        "zip_name": f"{stationxml_dir.name}.zip",
        "status": "PENDING",
        "bytes": 0,
    }


def discover_waveforms(waveforms_root: Path):
    for net_dir in sorted(waveforms_root.iterdir()):
        if not net_dir.is_dir():
            continue

        for leaf in sorted(net_dir.iterdir()):
            if not leaf.is_dir():
                continue

            yield {
                "type": "waveforms",
                "net": net_dir.name,
                "name": leaf.name,
                "local_path": str(leaf),
                "zip_name": f"{leaf.name}.zip",
                "status": "PENDING",
                "bytes": 0,
            }


# ----------------------------
# MAIN PIPELINE
# ----------------------------


def process_entry(
    entry: dict, tmp_zip_dir: Path, manifest_path: Path, dropbox_path: str
):
    src = Path(entry["local_path"])
    zip_path = zip_directory(src, tmp_zip_dir)

    if entry["type"] == "stationxml":
        dropbox_target = f"{dropbox_path}/stationxml"
    else:
        dropbox_target = f"{dropbox_path}/waveforms/{entry['net']}"

    try:
        ok = upload_and_verify(zip_path, dropbox_target)
    except Exception as e:
        print(f"ERROR uploading {zip_path.name}: {e}")
        update_manifest_status(manifest_path, entry["zip_name"], "FAILED", 0)
        return

    if ok:
        size = zip_path.stat().st_size
        update_manifest_status(manifest_path, entry["zip_name"], "DONE", size)
        zip_path.unlink()
        print(f"DONE {zip_path.name}")
    else:
        update_manifest_status(manifest_path, entry["zip_name"], "FAILED", 0)
        print(f"FAILED {zip_path.name}")


# ----------------------------
# CLI
# ----------------------------


@app.command()
def run(
    data_root: Path = typer.Argument(
        ..., help="Root directory containing waveforms/ and stationxml/"
    ),
    dropbox_path: str = typer.Argument(
        ...,
        help="Rclone Dropbox path to upload to.",
    ),
):
    """
    Resume-safe Dropbox backup with manifest tracking.
    """
    manifest = data_root / "dropbox_manifest.csv"
    manifest_rows = load_manifest(manifest)

    # Discover + register new entries
    stationxml = data_root / "stationxml"
    waveforms = data_root / "waveforms"

    tmp_zip_dir = data_root / "tmp_zips"
    tmp_zip_dir.mkdir(parents=True, exist_ok=True)

    for entry in discover_stationxml(stationxml):
        if entry["zip_name"] not in manifest_rows:
            append_manifest_row(manifest, entry)

    for entry in discover_waveforms(waveforms):
        if entry["zip_name"] not in manifest_rows:
            append_manifest_row(manifest, entry)

    # Reload after discovery
    manifest_rows = load_manifest(manifest)

    pending = [row for row in manifest_rows.values() if row["status"] != "DONE"]

    print(f"Pending uploads: {len(pending)}")

    for entry in pending:
        process_entry(entry, tmp_zip_dir, manifest, dropbox_path)


if __name__ == "__main__":
    app()
