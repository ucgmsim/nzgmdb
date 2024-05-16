"""
File that contains the function scripts that can be called to run the NZGMDB pipeline.
"""

from datetime import datetime
from pathlib import Path

import typer

from nzgmdb.data_retrieval.geonet import parse_geonet_information
from nzgmdb.data_retrieval.tect_domain import add_tect_domain
from nzgmdb.calculation.snr import compute_snr_for_mseed_data
from nzgmdb.management import file_structure
from nzgmdb.phase_arrival.gen_phase_arrival_table import (
    generate_phase_arrival_table,
)

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
def gen_phase_arrival_table(
    main_dir: Path, output_dir: Path, n_procs: int = 1, full_output: bool = False
):
    """
    Generate the phase arrival table, taking mseed data and finding the phase arrivals using a p_wave picker.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (highest level directory)
        (glob is used to find all mseed files recursively)
    output_dir : Path
        The directory to save the phase arrival table
    n_procs : int (optional)
        The number of processes to use to generate the phase arrival table
    full_output: bool (optional)
        If True, writes an additional table that
        contains all phase arrivals from both
        picker and Geonet
    """
    generate_phase_arrival_table(main_dir, output_dir, n_procs, full_output)


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
