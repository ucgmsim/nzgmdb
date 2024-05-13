"""
File that contains the function scripts that can be called to run the NZGMDB pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.calculation.snr import compute_snr_for_mseed_data
from nzgmdb.data_processing.process_observed import process_mseeds_to_txt
from nzgmdb.data_retrieval.geonet import parse_geonet_information
from nzgmdb.data_retrieval.tect_domain import add_tect_domain
from nzgmdb.management import file_structure
from nzgmdb.phase_arrival.gen_phase_arrival_table import (
    generate_phase_arrival_table,
)

app = typer.Typer()


@app.command(
    help="Fetch earthquake data from Geonet and generates the earthquake source and station magnitude tables."
)
def fetch_geonet_data(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
        ),
    ],
    start_date: Annotated[
        datetime,
        typer.Argument(
            help="The start date to filter the earthquake data",
        ),
    ],
    end_date: Annotated[
        datetime,
        typer.Argument(
            help="The end date to filter the earthquake data",
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
):
    parse_geonet_information(main_dir, start_date, end_date, n_procs)


@app.command(help="Add tectonic domains to the earthquake source table")
def merge_tect_domain(
    eq_source_ffp: Annotated[
        Path,
        typer.Argument(
            help="The file path to the earthquake source table",
            readable=True,
            exists=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the earthquake source table with tectonic domains",
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
):
    add_tect_domain(eq_source_ffp, output_dir, n_procs)


@app.command(
    help="Generate the phase arrival table, taking mseed data and finding the phase arrivals using a p_wave picker. "
    "Requires the mseed files to be generated."
)
def gen_phase_arrival_table(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory) "
            "(glob is used to find all mseed files recursively)",
            exists=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the phase arrival table",
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
):
    generate_phase_arrival_table(main_dir, output_dir, n_procs)


@app.command(
    help="Calculate the signal to noise ratio of the waveforms as well as FAS. "
    "Requires the phase arrival table and mseed files to be generated. "
    "Allows for custom links to the KO matrix, meta output directory, and SNR and FAS output directory. "
    "If not provided, the default directories are used as if running the full NZGMDB pipeline."
    "Note: Can't have the common frequency vector as an input due to typer limitations. "
    "Instead change the configuration file."
)
def calculate_snr(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
        ),
    ],
    phase_table_path: Annotated[
        Path,
        typer.Option(
            help="Path to the phase arrival table",
            readable=True,
            exists=True,
        ),
    ] = None,
    meta_output_dir: Annotated[
        Path,
        typer.Option(
            help="Path to the output directory for the metadata and skipped records",
        ),
    ] = None,
    snr_fas_output_dir: Annotated[
        Path,
        typer.Option(
            help="Path to the output directory for the SNR and FAS data",
        ),
    ] = None,
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
    apply_smoothing: Annotated[
        bool, typer.Option(help="Whether to apply smoothing to the SNR calculation")
    ] = True,
    ko_matrix_path: Annotated[
        Path,
        typer.Option(
            help="Path to the ko matrix",
            exists=True,
        ),
    ] = None,
):
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


@app.command(
    help="Process the mseed files to txt files. "
    "Saves the skipped records to a csv file and gives reasons why they were skipped"
)
def process_records(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
        ),
    ],
    gmc_ffp: Annotated[
        Path,
        typer.Option(
            help="The full file path to the GMC predictions file",
            readable=True,
            exists=True,
        ),
    ] = None,
    fmax_ffp: Annotated[
        Path,
        typer.Option(
            help="The full file path to the Fmax file",
            readable=True,
            exists=True,
        ),
    ] = None,
    n_procs: Annotated[int, typer.Option(help="The number of processes to use")] = 1,
):
    if gmc_ffp is None:
        gmc_ffp = file_structure.get_flatfile_dir(main_dir) / "gmc_predictions.csv"
    if fmax_ffp is None:
        fmax_ffp = file_structure.get_flatfile_dir(main_dir) / "fmax.csv"

    process_mseeds_to_txt(main_dir, gmc_ffp, fmax_ffp, n_procs)


@app.command(
    help="Run the first half of the NZGMDB pipeline before GMC. "
    "- Fetch Geonet data "
    "- Merge tectonic domains "
    "- Generate phase arrival table "
    "- Calculate SNR "
    "- Calculate Fmax"
)
def run_pre_process_nzgmdb(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
        ),
    ],
    start_date: Annotated[
        datetime,
        typer.Argument(
            help="The start date to filter the earthquake data",
        ),
    ],
    end_date: Annotated[
        datetime,
        typer.Argument(
            help="The end date to filter the earthquake data",
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="The number of processes to use")] = 1,
):
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


@app.command(
    help="Run the processing part of the NZGMDB pipeline, everything after GMC. "
    "- Process and filter waveform data to txt files "
    "- Calculate IM's "
    "- Merge IM results into flatfiles"
)
def run_process_nzgmdb(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="The number of processes to use")] = 1,
):
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Run filtering and processing of mseeds
    gmc_ffp = flatfile_dir / "gmc_predictions.csv"
    fmax_ffp = flatfile_dir / "fmax.csv"
    process_records(main_dir, gmc_ffp, fmax_ffp, n_procs)

    # Steps below are TODO
    # Run IM calculation
    # Merge flat files with IM results


if __name__ == "__main__":
    app()
