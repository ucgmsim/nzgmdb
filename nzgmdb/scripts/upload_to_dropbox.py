"""
Script to upload the NZGMDB results to Dropbox.
"""

import multiprocessing as mp
import subprocess
import zipfile
from pathlib import Path
from typing import Annotated, Optional

import typer

from nzgmdb.management import file_structure
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)

DROPBOX_PATH = "dropbox:/QuakeCoRE/Public/NZGMDB"


def zip_files(file_list: list, output_dir: Path, zip_name: str):
    """
    Zip specific files into one archive.

    Parameters
    ----------
    file_list : list
        List of files to zip
    output_dir : Path
        Directory to save the zip file
    zip_name : str
        Name of the zip file

    Returns
    -------
    Path
        The path to the zip file
    """
    zip_filename = output_dir / f"{zip_name}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_list:
            zipf.write(file_path, arcname=file_path.name)
    return zip_filename  # Return zip file path for later use


def upload_zip_to_dropbox(local_file: Path, dropbox_path: str):
    """
    Uploads a file to Dropbox using rclone.
    Also checks the file size to ensure it was uploaded correctly.

    Parameters
    ----------
    local_file : Path
        The local file to upload
    dropbox_path : str
        The path on Dropbox to upload the file to

    Returns
    -------
    Path | None
        The local file if it failed to upload, otherwise None
    """
    print(f"Uploading {local_file} to {dropbox_path}")
    try:
        subprocess.check_call(
            f"rclone --progress copy {local_file} {dropbox_path}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError:
        return local_file
    else:
        # Check file size is correct and uploaded successfully
        local_size = local_file.stat().st_size
        cmd = f"rclone lsf --format=s {dropbox_path}/{local_file.name}"
        p = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = p.communicate()
        output_decoded = out.decode("utf-8").strip()
        if not output_decoded:
            # File does not exist on Dropbox
            return local_file
        if int(output_decoded) != local_size:
            # File size does not match
            return local_file
        return None


def main(
    input_dir: Path,
    n_procs: int,
    version: str = None,
):
    """
    Main function to zip and upload all required files.

    Parameters
    ----------
    input_dir : Path
        The directory containing the NZGMDB results
    n_procs : int
        Number of processes to use
    version : str
        Version of the results, defaults to the directory name (used for folder name on dropbox)
    """
    output_dir = input_dir / "zips"
    output_dir.mkdir(parents=True, exist_ok=True)
    dropbox_version_dir = f"{DROPBOX_PATH}/{version}"

    if version is None:
        version = input_dir.name

    flatfiles_dir = file_structure.get_flatfile_dir(input_dir)
    snr_fas_dir = file_structure.get_snr_fas_dir(input_dir)
    waveforms_dir = file_structure.get_waveform_dir(input_dir)

    # 1) Zip the waveforms per year
    waveform_output_dir = output_dir / "waveforms"
    waveform_output_dir.mkdir(exist_ok=True)
    year_folders = [f for f in waveforms_dir.iterdir() if f.is_dir()]
    with mp.Pool(n_procs) as pool:
        waveforms_zip_files = pool.starmap(
            zip_files,
            [
                (list(folder.rglob("*.*")), waveform_output_dir, folder.stem)
                for folder in year_folders
            ],
        )
    # Also zip each event folder
    event_zips = {}
    for year_folder in year_folders:
        year_output_dir = waveform_output_dir / year_folder.name
        year_output_dir.mkdir(exist_ok=True)
        event_folders = [f for f in year_folder.iterdir() if f.is_dir()]
        with mp.Pool(n_procs) as pool:
            year_event_zips = pool.starmap(
                zip_files,
                [
                    (list(folder.rglob("*.*")), year_output_dir, folder.stem)
                    for folder in event_folders
                ],
            )
        event_zips[year_folder.name] = year_event_zips

    # 2) Zip flatfiles_{ver}.zip
    flatfiles = [flatfiles_dir / file for file in file_structure.FlatfileNames]
    flatfiles_zip = zip_files(flatfiles, output_dir, f"flatfiles_{version}")

    # Check if there is a quality_db directory and zip it
    quality_db_dir = input_dir / "quality_db"
    if quality_db_dir.exists():
        quality_db_files = list(quality_db_dir.rglob("*.csv"))
        quality_db_zip = zip_files(
            quality_db_files, output_dir, f"quality_flatfiles_{version}"
        )

        # Upload quality_db zip to Dropbox
        upload_zip_to_dropbox(quality_db_zip, dropbox_version_dir)

    # 3) Zip skipped_{ver}.zip
    skipped_files = [
        flatfiles_dir / file
        for file in file_structure.SkippedRecordFilenames
        if quality_db_dir.exists()
        or file != file_structure.SkippedRecordFilenames.QUALITY_SKIPPED_RECORDS
    ]
    skipped_zip = zip_files(skipped_files, output_dir, f"skipped_{version}")

    # 4) Zip pre_flatfiles_{ver}.zip
    pre_flatfiles = [flatfiles_dir / file for file in file_structure.PreFlatfileNames]
    pre_flatfiles_zip = zip_files(pre_flatfiles, output_dir, f"pre_flatfiles_{version}")

    # 5) Zip snr_fas_{ver}.zip
    snr_files = list(snr_fas_dir.rglob("*.csv"))
    snr_fas_zip = zip_files(snr_files, output_dir, f"snr_fas_{version}")

    # Upload everything to Dropbox
    dropbox_waveforms_path = f"{dropbox_version_dir}/waveforms"
    # Upload waveform year zips
    with mp.Pool(n_procs) as pool:
        failed_files = pool.starmap(
            upload_zip_to_dropbox,
            [(zip_file, dropbox_waveforms_path) for zip_file in waveforms_zip_files],
        )
    # Upload event zips
    for year, event_zips in event_zips.items():
        dropbox_year_path = f"{dropbox_waveforms_path}/{year}"
        with mp.Pool(n_procs) as pool:
            failed_files.extend(
                pool.starmap(
                    upload_zip_to_dropbox,
                    [(zip_file, dropbox_year_path) for zip_file in event_zips],
                )
            )

    failed_files.append(upload_zip_to_dropbox(flatfiles_zip, dropbox_version_dir))
    failed_files.append(upload_zip_to_dropbox(skipped_zip, dropbox_version_dir))
    failed_files.append(upload_zip_to_dropbox(pre_flatfiles_zip, dropbox_version_dir))
    failed_files.append(upload_zip_to_dropbox(snr_fas_zip, dropbox_version_dir))

    # Remove any None values from the failed_files list
    failed_files = [f for f in failed_files if f is not None]

    if failed_files:
        # Save the failed files to a file
        failed_files_file = output_dir / "failed_files.txt"
        with open(failed_files_file, "w") as f:
            f.write("\n".join([str(f) for f in failed_files]))
        print(
            f"Failed to upload {len(failed_files)} files. See {failed_files_file} for paths."
        )
    else:
        print("All files uploaded successfully.")


def determine_dropbox_path(dropbox_version_dir: str, local_file: Path):
    """
    Helper function to determine the dropbox path for a local failed file.

    Parameters
    ----------
    dropbox_version_dir : str
        The version directory on Dropbox
    local_file : Path
        The local file that failed to upload

    Returns
    -------
    str
        The path on Dropbox to upload the file to
    """
    parts = local_file.parts
    if "zips" in parts:
        zips_index = parts.index("zips")
        relative_path = "/".join(parts[zips_index + 1 :])
        return f"{dropbox_version_dir}/{relative_path}"
    else:
        return dropbox_version_dir


@cli.from_docstring(app)
def upload_to_dropbox(
    input_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    version: Annotated[Optional[str], typer.Option()] = None,
    n_procs: Annotated[int, typer.Option()] = 1,
) -> None:
    """Upload results to Dropbox.

    Parameters
    ----------
    input_directory : Path
        Directory containing the results.
    version : Optional[str]
        Version of the results, defaults to the directory name.
    n_procs : int
        Number of processes to use.
    """
    main(
        input_directory,
        n_procs,
        version if version is not None else input_directory.name,
    )


@cli.from_docstring(app)
def upload_failed_files(
    failed_files_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    version: Annotated[Optional[str], typer.Option()] = None,
    n_procs: Annotated[int, typer.Option()] = 1,
) -> None:
    """Upload failed files.

    Parameters
    ----------
    failed_files_file : Path
        File containing the failed files.
    version : Optional[str]
        Version of the results, defaults to the directory name.
    n_procs : int
        Number of processes to use.
    """
    with open(failed_files_file, "r") as f:
        failed_files = f.read().splitlines()

    if version is None:
        version = failed_files_file.parent.name

    dropbox_version_dir = f"{DROPBOX_PATH}/{version}"

    with mp.Pool(n_procs) as pool:
        failed_files = pool.starmap(
            upload_zip_to_dropbox,
            [
                (Path(f), determine_dropbox_path(dropbox_version_dir, Path(f)))
                for f in failed_files
            ],
        )

    failed_files = [f for f in failed_files if f is not None]
    if failed_files:
        # Save the failed files to a file
        failed_files_file = (
            failed_files_file.parent / f"{failed_files_file.stem}_rerun.txt"
        )
        with open(failed_files_file, "w") as f:
            f.write("\n".join([str(f) for f in failed_files]))
        print(
            f"Failed to upload {len(failed_files)} files. See {failed_files_file.stem}_rerun.txt for paths."
        )
    else:
        print("All files uploaded successfully.")


if __name__ == "__main__":
    app()
