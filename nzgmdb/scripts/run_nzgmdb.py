"""
File that contains the function scripts that can be called to run the NZGMDB pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.calculation import fmax, ims, snr, distances
from nzgmdb.data_processing import process_observed, merge_flatfiles
from nzgmdb.data_retrieval import geonet, tect_domain, sites
from nzgmdb.management import file_structure
from nzgmdb.phase_arrival import gen_phase_arrival_table

app = typer.Typer()


@app.command(
    help="Fetch earthquake data from Geonet and generates the earthquake source and station magnitude tables."
)
def fetch_geonet_data(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            file_okay=False,
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
    geonet.parse_geonet_information(main_dir, start_date, end_date, n_procs)


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
            file_okay=False,
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
):
    tect_domain.add_tect_domain(eq_source_ffp, output_dir, n_procs)


@app.command(
    help="Generate the phase arrival table, taking mseed data and finding the phase arrivals using a p_wave picker. "
    "Requires the mseed files to be generated."
)
def make_phase_arrival_table(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory) "
            "(glob is used to find all mseed files recursively)",
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the phase arrival table", file_okay=False
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
    full_output: Annotated[
        bool, typer.Option(help="output arrival times from GeoNet and Picker")
    ] = False,
):
    gen_phase_arrival_table.generate_phase_arrival_table(
        main_dir, output_dir, n_procs, full_output
    )


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
            file_okay=False,
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
            file_okay=False,
        ),
    ] = None,
    snr_fas_output_dir: Annotated[
        Path,
        typer.Option(
            help="Path to the output directory for the SNR and FAS data",
            file_okay=False,
        ),
    ] = None,
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
    apply_smoothing: Annotated[
        bool, typer.Option(help="Whether to apply smoothing to the SNR calculation")
    ] = True,
    ko_matrix_path: Annotated[
        Path,
        typer.Option(
            help="Path to the ko matrix directory",
            exists=True,
            file_okay=False,
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
    snr.compute_snr_for_mseed_data(
        main_dir,
        phase_table_path,
        meta_output_dir,
        snr_fas_output_dir,
        n_procs,
        apply_smoothing,
        ko_matrix_path,
    )


@app.command(
    help="Calculate the maximum useable frequency (fmax). "
    "Requires the snr_fas files and the snr metadata. "
    "Several parameters are set in the config file."
)
def calc_fmax(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
            file_okay=False,
        ),
    ],
    meta_output_dir: Annotated[
        Path,
        typer.Option(
            help="Path to the output directory for the metadata and skipped records",
            file_okay=False,
        ),
    ] = None,
    snr_fas_output_dir: Annotated[
        Path,
        typer.Option(
            help="Path to the output directory for the SNR and FAS data",
            file_okay=False,
        ),
    ] = None,
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
):
    if meta_output_dir is None:
        meta_output_dir = file_structure.get_flatfile_dir(main_dir)
    if snr_fas_output_dir is None:
        snr_fas_output_dir = file_structure.get_snr_fas_dir(main_dir)

    fmax.run_full_fmax_calc(meta_output_dir, snr_fas_output_dir, n_procs)


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
            file_okay=False,
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

    process_observed.process_mseeds_to_txt(main_dir, gmc_ffp, fmax_ffp, n_procs)


@app.command(help="Run IM Calculation on processed waveform files")
def run_im_calculation(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="The directory to save the IM files", file_okay=False),
    ] = None,
    n_procs: Annotated[int, typer.Option(help="The number of processes to use")] = 1,
    checkpoint: Annotated[
        bool,
        typer.Option(
            help="If True, the function will check for already completed files and skip them"
        ),
    ] = False,
):
    if output_dir is None:
        output_dir = file_structure.get_im_dir(main_dir)
    ims.compute_ims_for_all_processed_records(main_dir, output_dir, n_procs, checkpoint)


@app.command(help="Generate the site table basin flatfile")
def generate_site_table_basin(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
            file_okay=False,
        ),
    ],
):
    site_df = sites.create_site_table_response()
    site_df = sites.add_site_basins(site_df)
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    site_df.to_csv(flatfile_dir / "site_table_basin.csv", index=False)


@app.command(
    help="Calculate the distances between the earthquake source and the station"
)
def calculate_distances(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
            file_okay=False,
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="The number of processes to use")] = 1,
):
    distances.calc_distances(main_dir, n_procs)


@app.command(
    help="Merge IM results together into one flatfile. As well as perform a filter for Ds595"
)
def merge_im_results(
    im_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory containing the IM results to merge",
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the merged IM file", file_okay=False
        ),
    ],
    gmc_ffp: Annotated[
        Path,
        typer.Argument(
            help="The full file path to the GMC predictions file",
            readable=True,
            exists=True,
        ),
    ],
    fmax_ffp: Annotated[
        Path,
        typer.Argument(
            help="The full file path to the Fmax file",
            readable=True,
            exists=True,
        ),
    ],
):
    merge_flatfiles.merge_im_data(im_dir, output_dir, gmc_ffp, fmax_ffp)


@app.command(
    help="Merge all flatfiles together for final output and ensure correct filtering for only results with IM values"
)
def merge_flat_files(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
            file_okay=False,
        ),
    ],
):
    merge_flatfiles.merge_flatfiles(main_dir)


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
            file_okay=False,
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
    geonet.parse_geonet_information(main_dir, start_date, end_date, n_procs)

    # Merge the tectonic domains
    faltfile_dir = file_structure.get_flatfile_dir(main_dir)
    eq_source_ffp = faltfile_dir / "earthquake_source_table.csv"
    eq_tect_domain_ffp = faltfile_dir / "earthquake_source_table_tectdomain.csv"
    tect_domain.add_tect_domain(eq_source_ffp, eq_tect_domain_ffp, n_procs)

    # Generate the phase arrival table
    gen_phase_arrival_table.generate_phase_arrival_table(
        main_dir, faltfile_dir, n_procs
    )

    # Generate SNR
    meta_output_dir = file_structure.get_flatfile_dir(main_dir)
    snr_fas_output_dir = file_structure.get_snr_fas_dir(main_dir)
    phase_table_path = (
        file_structure.get_flatfile_dir(main_dir) / "phase_arrival_table.csv"
    )
    calculate_snr(
        main_dir, phase_table_path, meta_output_dir, snr_fas_output_dir, n_procs
    )

    # Calculate Fmax
    calc_fmax(main_dir, meta_output_dir, snr_fas_output_dir, n_procs)


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

    # Run IM calculation
    im_dir = file_structure.get_im_dir(main_dir)
    run_im_calculation(main_dir, im_dir, n_procs)

    # Merge IM results
    merge_im_results(im_dir, flatfile_dir, gmc_ffp, fmax_ffp)

    # Generate the site basin flatfile
    generate_site_table_basin(main_dir)

    # Calculate distances
    distances.calc_distances(main_dir, n_procs)
    
    # Merge flat files
    merge_flat_files(main_dir)


if __name__ == "__main__":
    app()
