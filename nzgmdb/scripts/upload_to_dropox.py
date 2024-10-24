import subprocess
import multiprocessing as mp
import os
import zipfile
import typer
from pathlib import Path

from nzgmdb.management import file_structure

app = typer.Typer()

DROPBOX_PATH = "dropbox:/QuakeCoRE/Public/NZGMDB"


def zip_folder(folder_path: Path, output_dir: Path, zip_name: str):
    """Zips the contents of a folder."""
    zip_filename = output_dir / f"{zip_name}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = Path(root) / file
                zipf.write(file_path, file_path.relative_to(folder_path.parent))
    return zip_filename  # Return zip file path for later use


def zip_files(file_list, output_dir: Path, zip_name: str):
    """Zip specific files into one archive."""
    zip_filename = output_dir / f"{zip_name}.zip"
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in file_list:
            zipf.write(file_path, arcname=file_path.name)
    return zip_filename  # Return zip file path for later use


def upload_zip_to_dropbox(local_file: Path, dropbox_path: str):
    """Uploads a file to Dropbox using rclone."""
    print(f"Uploading {local_file} to {dropbox_path}")
    p = subprocess.Popen(
        f"rclone --progress copy {local_file} {dropbox_path}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print(f"Error uploading {local_file}: {stderr}")
    else:
        print(f"Successfully uploaded {local_file}")


def main(
    input_dir: Path,
    n_procs: int,
    version: str,
):
    """Main function to zip and upload all required files."""
    output_dir = input_dir / "zips"
    output_dir.mkdir(parents=True, exist_ok=True)

    flatfiles_dir = file_structure.get_flatfile_dir(input_dir)
    snr_fas_dir = file_structure.get_snr_fas_dir(input_dir)
    waveforms_dir = file_structure.get_waveform_dir(input_dir)

    # 1) Zip the waveforms per year
    year_folders = [f for f in waveforms_dir.iterdir() if f.is_dir()]
    with mp.Pool(n_procs) as pool:
        waveforms_zip_files = pool.starmap(
            zip_folder, [(folder, output_dir, folder.stem) for folder in year_folders]
        )

    # 2) Zip flatfiles_{ver}.zip
    flatfiles = [flatfiles_dir / file for file in file_structure.FlatfileNames]
    flatfiles_zip = zip_files(flatfiles, output_dir, f"flatfiles_{version}")

    # 3) Zip skipped_{ver}.zip
    skipped_files = [
        flatfiles_dir / file for file in file_structure.SkippedRecordFilenames
    ]
    skipped_zip = zip_files(skipped_files, output_dir, f"skipped_{version}")

    # 4) Zip pre_flatfiles_{ver}.zip
    pre_flatfiles = [flatfiles_dir / file for file in file_structure.PreFlatfileNames]
    pre_flatfiles_zip = zip_files(pre_flatfiles, output_dir, f"pre_flatfiles_{version}")

    # 5) Zip snr_fas_{ver}.zip
    snr_fas_zip = zip_folder(snr_fas_dir, output_dir, f"snr_fas_{version}")

    # Upload everything to Dropbox
    dropbox_version_dir = f"{DROPBOX_PATH}/{version}"
    dropbox_waveforms_path = f"{dropbox_version_dir}/waveforms"
    for zip_file in waveforms_zip_files:
        upload_zip_to_dropbox(zip_file, dropbox_waveforms_path)

    upload_zip_to_dropbox(flatfiles_zip, dropbox_version_dir)
    upload_zip_to_dropbox(skipped_zip, dropbox_version_dir)
    upload_zip_to_dropbox(pre_flatfiles_zip, dropbox_version_dir)
    upload_zip_to_dropbox(snr_fas_zip, dropbox_version_dir)


@app.command()
def upload_to_dropbox(
    input_directory: Path = typer.Argument(
        ..., help="Directory containing the results"
    ),
    version: str = typer.Argument(..., help="Version of the results"),
    num_processes: int = typer.Option(6, help="Number of processes to use"),
):
    main(
        input_directory,
        num_processes,
        version,
    )


if __name__ == "__main__":
    app()
