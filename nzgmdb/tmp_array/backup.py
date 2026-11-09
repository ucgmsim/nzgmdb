"""
Back up temporary array data to Dropbox, one zip per station folder, tracked in a manifest.

A folder is uploaded again when its file count or size changes, so data added by
later download windows is also backed up.
"""

import zipfile
from pathlib import Path

import pandas as pd

from nzgmdb.scripts import upload_to_dropbox
from nzgmdb.tmp_array import file_layout

MANIFEST_NAME = "dropbox_manifest.csv"
MANIFEST_COLUMNS = [
    "dropbox_dir",
    "local_path",
    "zip_name",
    "n_files",
    "n_bytes",
    "status",
]


def folder_fingerprint(folder: Path) -> tuple[int, int]:
    """
    Get the file count and total size of a folder.

    Parameters
    ----------
    folder : Path
        Folder to measure

    Returns
    -------
    tuple[int, int]
        (n_files, n_bytes)
    """
    files = [f for f in folder.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def find_backup_entries(data_dir: Path, dropbox_path: str) -> pd.DataFrame:
    """
    List the folders to back up: one per waveform station folder plus the whole StationXML folder.

    Parameters
    ----------
    data_dir : Path
        Root temporary array data directory
    dropbox_path : str
        Base rclone Dropbox path

    Returns
    -------
    pd.DataFrame
        One row per folder with its current fingerprint
    """
    folders = []
    stationxml_dir = data_dir / file_layout.STATIONXML_DIRNAME
    if stationxml_dir.is_dir():
        folders.append(
            (stationxml_dir, f"{dropbox_path}/{file_layout.STATIONXML_DIRNAME}")
        )
    for net_dir in sorted((data_dir / file_layout.WAVEFORM_DIRNAME).glob("*")):
        for station_dir in sorted(p for p in net_dir.iterdir() if p.is_dir()):
            folders.append(
                (
                    station_dir,
                    f"{dropbox_path}/{file_layout.WAVEFORM_DIRNAME}/{net_dir.name}",
                )
            )

    rows = [
        [
            dropbox_dir,
            str(folder),
            f"{folder.name}.zip",
            *folder_fingerprint(folder),
            "",
        ]
        for folder, dropbox_dir in folders
    ]
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def zip_folder(folder: Path, zip_ffp: Path) -> Path:
    """
    Zip a folder, storing files uncompressed as mseed is already compressed.

    Parameters
    ----------
    folder : Path
        Folder to zip
    zip_ffp : Path
        Output zip file path

    Returns
    -------
    Path
        The zip file path
    """
    with zipfile.ZipFile(zip_ffp, "w", zipfile.ZIP_STORED) as zipf:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                zipf.write(f, arcname=f.relative_to(folder.parent))
    return zip_ffp


def backup_to_dropbox(data_dir: Path, dropbox_path: str) -> pd.DataFrame:
    """
    Zip and upload new or changed folders, updating the manifest after each upload.

    Parameters
    ----------
    data_dir : Path
        Root temporary array data directory
    dropbox_path : str
        Base rclone Dropbox path

    Returns
    -------
    pd.DataFrame
        The updated manifest
    """
    manifest_ffp = data_dir / MANIFEST_NAME
    manifest_df = find_backup_entries(data_dir, dropbox_path)

    # Keep DONE only when the folder has not changed since it was uploaded
    if manifest_ffp.exists():
        old_df = pd.read_csv(manifest_ffp, keep_default_na=False)
        if "n_files" not in old_df.columns:
            # Manifest from the old backup script has no fingerprint, trust its DONE rows as they are now
            old_df = old_df[["local_path", "status"]].merge(
                manifest_df[["local_path", "n_files", "n_bytes"]], on="local_path"
            )
        manifest_df = manifest_df.merge(
            old_df[["local_path", "n_files", "n_bytes", "status"]],
            on=["local_path", "n_files", "n_bytes"],
            how="left",
            suffixes=("_new", ""),
        ).drop(columns="status_new")
        manifest_df["status"] = manifest_df["status"].fillna("")
    manifest_df.loc[manifest_df["status"] != "DONE", "status"] = "PENDING"
    manifest_df.to_csv(manifest_ffp, index=False)

    pending = manifest_df.index[manifest_df["status"] == "PENDING"]
    print(f"{len(pending)} of {len(manifest_df)} folders to upload")

    tmp_zip_dir = data_dir / "tmp_zips"
    tmp_zip_dir.mkdir(exist_ok=True)
    for idx in pending:
        entry = manifest_df.loc[idx]
        zip_ffp = zip_folder(Path(entry["local_path"]), tmp_zip_dir / entry["zip_name"])
        failed = upload_to_dropbox.upload_zip_to_dropbox(zip_ffp, entry["dropbox_dir"])
        manifest_df.loc[idx, "status"] = "FAILED" if failed else "DONE"
        manifest_df.to_csv(manifest_ffp, index=False)
        zip_ffp.unlink()
        print(f"{manifest_df.loc[idx, 'status']} {entry['zip_name']}")

    return manifest_df
