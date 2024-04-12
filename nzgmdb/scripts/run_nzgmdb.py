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
from nzgmdb.management import file_structure

app = typer.Typer()


@app.command()
def gen_phase_arrival_table(main_dir: Path, output_dir: Path, n_procs: int = 1):
    """
    Generate the phase arrival table, taking mseed data and finding the phase arrivals using a p_wave picker.

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
    add_tect_domain(eq_source_ffp, faltfile_dir, n_procs)

    # Generate the phase arrival table
    generate_phase_arrival_table(main_dir, faltfile_dir, n_procs)

    # Steps below are TODO
    # Generate SNR
    # Calculate Fmax
    # Run filtering and processing of mseeds
    # Run IM calculation
    # Merge flat files with IM results


if __name__ == "__main__":
    app()
