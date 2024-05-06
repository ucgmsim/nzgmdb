"""
File that contains the function scripts that can be called to run the NZGMDB pipeline.
"""

import typer
from datetime import datetime
from pathlib import Path

from nzgmdb.phase_arrival.gen_phase_arrival_table import (
    generate_phase_arrival_table,
)
from nzgmdb.data_retrieval.geonet import parse_geonet_information
from nzgmdb.data_retrieval.tect_domain import add_tect_domain
from nzgmdb.calculation.snr import compute_snr_for_mseed_data
from nzgmdb.management import file_structure

app = typer.Typer()


@app.command()
def fetch_geonet_data(
    main_dir: Path,
    start_date: datetime,
    end_date: datetime,
    n_procs: int = 1,
):
    """
    Fetches earthquake data from Geonet and generates the earthquake source and station magnitude tables.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    start_date : datetime
        The start date to filter the earthquake data
    end_date : datetime
        The end date to filter the earthquake data
    n_procs : int (optional)
        The number of processes to use to generate the earthquake source and station magnitude tables
    """
    parse_geonet_information(main_dir, start_date, end_date, n_procs)


@app.command()
def merge_tect_domain(
    eq_source_ffp: Path,
    output_dir: Path,
    n_procs: int = 1,
):
    """
    Adds tectonic domains to the earthquake source table.

    Parameters
    ----------
    eq_source_ffp : Path
        The file path to the earthquake source table
    output_dir : Path
        The directory to save the earthquake source table with tectonic domains
    n_procs : int (optional)
        The number of processes to use for processing
    """
    add_tect_domain(eq_source_ffp, output_dir, n_procs)


@app.command()
def gen_phase_arrival_table(main_dir: Path, output_dir: Path, n_procs: int = 1):
    """
    Generate the phase arrival table, taking mseed data and finding the phase arrivals using a p_wave picker.
    Requires the mseed files to be generated.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
        (glob is used to find all mseed files recursively)
    output_dir : Path
        The directory to save the phase arrival table
    n_procs : int (optional)
        The number of processes to use to generate the phase arrival table
    """
    generate_phase_arrival_table(main_dir, output_dir, n_procs)


@app.command()
def calculate_snr(
    main_dir: Path,
    phase_table_path: Path = None,
    meta_output_dir: Path = None,
    snr_fas_output_dir: Path = None,
    n_procs: int = 1,
    apply_smoothing: bool = True,
    ko_matrix_path: Path = None,
):
    """
    Calculate the signal to noise ratio of the waveforms as well as FAS.
    Requires the phase arrival table and mseed files to be generated.
    Allows for custom links to the KO matrix, meta output directory, and SNR and FAS output directory.
    If not provided, the default directories are used as if running the full NZGMDB pipeline.

    Note: Can't have the common frequency vector as an input due to typer limitations.
    Instead change the configuration file.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    phase_table_path : Path
        Path to the phase arrival table
    meta_output_dir : Path
        Path to the output directory for the metadata and skipped records
    snr_fas_output_dir : Path
        Path to the output directory for the SNR and FAS data
    n_procs : int, optional
        Number of processes to use, by default 1
    apply_smoothing : bool, optional
        Whether to apply smoothing to the SNR calculation, by default True
    ko_matrix_path : Path, optional
        Path to the ko matrix, by default None
    """
    # Define the default paths if not provided
    if phase_table_path is None:
        phase_table_path = (
            file_structure.get_flatfile_dir(main_dir) / "phase_arrival_table.csv"
        )
    if meta_output_dir is None:
        meta_output_dir = file_structure.get_flatfile_dir(main_dir)
    if snr_fas_output_dir is None:
        snr_fas_output_dir = file_structure.get_snr_fas_dir(main_dir)
    compute_snr_for_mseed_data(
        main_dir,
        phase_table_path,
        meta_output_dir,
        snr_fas_output_dir,
        n_procs,
        apply_smoothing,
        ko_matrix_path,
    )


@app.command()
def run_full_nzgmdb(
    main_dir: Path, start_date: datetime, end_date: datetime, n_procs: int = 1
):
    """
    Run the full NZGMDB pipeline.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    start_date : datetime
        The start date to filter the earthquake data
    end_date : datetime
        The end date to filter the earthquake data
    n_procs : int (optional)
        The number of processes to use for processing
    """
    # Fetch the Geonet data
    parse_geonet_information(main_dir, start_date, end_date, n_procs)

    # Merge the tectonic domains
    faltfile_dir = file_structure.get_flatfile_dir(main_dir)
    eq_source_ffp = faltfile_dir / "earthquake_source_table.csv"
    eq_tect_domain_ffp = faltfile_dir / "earthquake_source_table_tectdomain.csv"
    add_tect_domain(eq_source_ffp, eq_tect_domain_ffp, n_procs)

    # Generate the phase arrival table
    generate_phase_arrival_table(main_dir, faltfile_dir, n_procs)

    # Generate SNR
    meta_output_dir = file_structure.get_flatfile_dir(main_dir)
    snr_fas_output_dir = file_structure.get_snr_fas_dir(main_dir)
    phase_table_path = (
        file_structure.get_flatfile_dir(main_dir) / "phase_arrival_table.csv"
    )
    calculate_snr(
        main_dir, phase_table_path, meta_output_dir, snr_fas_output_dir, n_procs
    )

    # Steps below are TODO
    # Calculate Fmax
    # Run filtering and processing of mseeds
    # Run IM calculation
    # Merge flat files with IM results


if __name__ == "__main__":
    app()
