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

app = typer.Typer()


@app.command()
def gen_phase_arrival_table(data_dir: Path, output_dir: Path, n_procs: int):
    """
    Generate the phase arrival table, taking mseed data and finding the phase arrivals using a p_wave picker.

    Parameters
    ----------
    data_dir : Path
        The top directory containing the mseed files
        (glob is used to find all mseed files recursively)
    output_dir : Path
        The directory to save the phase arrival table
    n_procs : int
        The number of processes to use to generate the phase arrival table
    """
    generate_phase_arrival_table(data_dir, output_dir, n_procs)


@app.command()
def fetch_geonet_data(
    earthquake_ffp: Path,
    output_dir: Path,
    start_date: datetime,
    end_date: datetime,
    n_procs: int,
):
    """
    Fetches earthquake data from Geonet and generates the earthquake source and station magnitude tables.

    Parameters
    ----------
    earthquake_ffp : Path
        The file path to the earthquake data directly downloaded from geonet
    output_dir : Path
        The directory to save the earthquake source and station magnitude tables
    start_date : datetime
        The start date to filter the earthquake data
    end_date : datetime
        The end date to filter the earthquake data
    n_procs : int
        The number of processes to use to generate the earthquake source and station magnitude tables
    """
    parse_geonet_information(earthquake_ffp, output_dir, start_date, end_date, n_procs)


if __name__ == "__main__":
    app()
