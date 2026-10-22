import csv
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

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


class ManifestRow(TypedDict):
    """A single row in the Dropbox backup manifest CSV."""

    type: str
    net: str
    name: str
    local_path: str
    zip_name: str
    status: str
    bytes: int


def zip_directory(src_dir: Path, out_dir: Path) -> Path:
    """Create a zip file for a directory.

    This uses the system `zip` executable (fast + reliable) and skips work if the
    expected zip already exists.

    Parameters
    ----------
    src_dir : Path
        Source directory to zip.
    out_dir : Path
        Output directory where the zip file will be written.

    Returns
    -------
    Path
        Path to the created (or existing) zip file.
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
    """Upload a file to Dropbox using rclone and verify by file size.

    Parameters
    ----------
    local_file : Path
        File to upload.
    dropbox_dir : str
        Rclone remote path (directory) to upload into.

    Returns
    -------
    bool
        True if the remote file exists and its size matches the local file.
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


def load_manifest(path: Path) -> dict[str, ManifestRow]:
    """Load an upload manifest keyed by ``zip_name``.

    Parameters
    ----------
    path : Path
        Path to the manifest CSV.

    Returns
    -------
    dict[str, ManifestRow]
        Manifest rows keyed by ``zip_name``. Returns an empty dict if the file
        does not exist.
    """
    rows: dict[str, ManifestRow] = {}

    if not path.exists():
        return rows

    with path.open(newline="") as f:
        reader: csv.DictReader[str] = csv.DictReader(f)
        for row in reader:
            # DictReader gives us strings; coerce the known int column.
            bytes_value = int(row.get("bytes") or 0)
            zip_name = row.get("zip_name")
            if not zip_name:
                # Keep behavior conservative: skip malformed rows.
                continue

            rows[zip_name] = ManifestRow(
                type=row.get("type", ""),
                net=row.get("net", ""),
                name=row.get("name", ""),
                local_path=row.get("local_path", ""),
                zip_name=zip_name,
                status=row.get("status", ""),
                bytes=bytes_value,
            )

    return rows


def append_manifest_row(path: Path, row: ManifestRow) -> None:
    """Append a single row to the manifest CSV, creating it if needed.

    Parameters
    ----------
    path : Path
        Path to the manifest CSV.
    row : ManifestRow
        The manifest row to append.

    Returns
    -------
    None
    """
    new_file = not path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def update_manifest_status(path: Path, zip_name: str, status: str, size: int) -> None:
    """Update an existing manifest row with a new status and size.

    Parameters
    ----------
    path : Path
        Path to the manifest CSV.
    zip_name : str
        Zip filename key for the entry to update.
    status : str
        New status string (e.g. "DONE", "FAILED").
    size : int
        Size of the zip file in bytes.
    """
    rows = load_manifest(path)
    rows[zip_name]["status"] = status
    rows[zip_name]["bytes"] = size

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(rows.values())


def discover_stationxml(stationxml_dir: Path) -> Iterator[ManifestRow]:
    """Discover the StationXML directory entry.

    Parameters
    ----------
    stationxml_dir : Path
        Path to the directory containing StationXML files.

    Returns
    -------
    Iterator[ManifestRow]
        An iterator yielding a single manifest row representing the StationXML
        directory.
    """
    yield ManifestRow(
        type="stationxml",
        net="",
        name=stationxml_dir.name,
        local_path=str(stationxml_dir),
        zip_name=f"{stationxml_dir.name}.zip",
        status="PENDING",
        bytes=0,
    )


def discover_waveforms(waveforms_root: Path) -> Iterator[ManifestRow]:
    """Discover waveform leaf directories to back up.

    The expected directory structure is ``waveforms/<net>/<leaf>/`` where each
    ``leaf`` directory is zipped independently.

    Parameters
    ----------
    waveforms_root : Path
        Root directory containing per-network waveform directories.

    Returns
    -------
    Iterator[ManifestRow]
        Iterator of manifest rows for each leaf directory.
    """
    for net_dir in sorted(waveforms_root.iterdir()):
        if not net_dir.is_dir():
            continue

        for leaf in sorted(net_dir.iterdir()):
            if not leaf.is_dir():
                continue

            yield ManifestRow(
                type="waveforms",
                net=net_dir.name,
                name=leaf.name,
                local_path=str(leaf),
                zip_name=f"{leaf.name}.zip",
                status="PENDING",
                bytes=0,
            )


def process_entry(
    entry: ManifestRow, tmp_zip_dir: Path, manifest_path: Path, dropbox_path: str
) -> None:
    """Zip, upload, verify, and update manifest for a single entry.

    Parameters
    ----------
    entry : ManifestRow
        Manifest entry describing what to back up.
    tmp_zip_dir : Path
        Directory where zips are created temporarily.
    manifest_path : Path
        Path to the manifest CSV.
    dropbox_path : str
        Base rclone Dropbox remote path.
    """
    src = Path(entry["local_path"])
    zip_path = zip_directory(src, tmp_zip_dir)

    if entry["type"] == "stationxml":
        dropbox_target = f"{dropbox_path}/stationxml"
    else:
        dropbox_target = f"{dropbox_path}/waveforms/{entry['net']}"

    try:
        ok = upload_and_verify(zip_path, dropbox_target)
    except (OSError, subprocess.CalledProcessError) as e:
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


@app.command()
def run(
    data_root: Path = typer.Argument(
        ..., help="Root directory containing waveforms/ and stationxml/"
    ),
    dropbox_path: str = typer.Argument(
        ...,
        help="Rclone Dropbox path to upload to.",
    ),
) -> None:
    """Resume-safe Dropbox backup with manifest tracking.

    Parameters
    ----------
    data_root : Path
        Root directory containing ``waveforms/`` and ``stationxml/``.
    dropbox_path : str
        Base rclone Dropbox remote path to upload to.
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
