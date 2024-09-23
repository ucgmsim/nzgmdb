"""
File that contains the function scripts that can be called to run the NZGMDB pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated, List

import typer

from nzgmdb.calculation import distances, fmax, ims, snr
from nzgmdb.data_processing import merge_flatfiles, process_observed
from nzgmdb.data_retrieval import geonet, sites, tect_domain
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
    batch_size: Annotated[
        int,
        typer.Option(
            help="The batch size for the Geonet data retrieval for how many events to process at a time",
        ),
    ] = 500,
    only_event_ids: Annotated[
        List[str],
        typer.Option(
            help="A list of event ids to filter the earthquake data, separated by commas",
            callback=lambda x: [] if x is None else x[0].split(","),
        ),
    ] = None,
    only_sites: Annotated[
        List[str],
        typer.Option(
            help="A list of site names to filter the earthquake data, separated by commas",
            callback=lambda x: [] if x is None else x[0].split(","),
        ),
    ] = None,
):
    geonet.parse_geonet_information(
        main_dir, start_date, end_date, n_procs, batch_size, only_event_ids, only_sites
    )


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
    no_smoothing: Annotated[
        bool,
        typer.Option(
            help="Enable to disable smoothing to the SNR calculation", is_flag=True
        ),
    ] = False,
    ko_matrix_path: Annotated[
        Path,
        typer.Option(
            help="Path to the ko matrix directory",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            help="The batch size for the SNR calculation for how many mseeds to process at a time",
        ),
    ] = 5000,
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
        apply_smoothing=not no_smoothing,
        ko_matrix_path=ko_matrix_path,
        batch_size=batch_size,
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
            help="If True, the function will check for already completed files and skip them",
            is_flag=True,
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
    main_dir.mkdir(parents=True, exist_ok=True)
    # Generate the site basin flatfile
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    flatfile_dir.mkdir(parents=True, exist_ok=True)

    site_df = sites.create_site_table_response()
    site_df = sites.add_site_basins(site_df)

    site_df.to_csv(flatfile_dir / "site_table.csv", index=False)


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
    no_smoothing: Annotated[
        bool,
        typer.Option(
            help="Enable to disable smoothing to the SNR calculation", is_flag=True
        ),
    ] = False,
    ko_matrix_path: Annotated[
        Path,
        typer.Option(
            help="Path to the ko matrix directory",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    only_event_ids: Annotated[
        List[str],
        typer.Option(
            help="A list of event ids to filter the earthquake data, separated by commas",
            callback=lambda x: [] if x is None else x[0].split(","),
        ),
    ] = None,
    only_sites: Annotated[
        List[str],
        typer.Option(
            help="A list of site names to filter the earthquake data, separated by commas",
            callback=lambda x: [] if x is None else x[0].split(","),
        ),
    ] = None,
    geonet_batch_size: Annotated[
        int,
        typer.Option(
            help="The batch size for the Geonet data retrieval for how many events to process at a time",
        ),
    ] = 500,
    snr_batch_size: Annotated[
        int,
        typer.Option(
            help="The batch size for the SNR calculation for how many mseeds to process at a time",
        ),
    ] = 5000,
):
    main_dir.mkdir(parents=True, exist_ok=True)

    # Generate the site basin flatfile
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    flatfile_dir.mkdir(parents=True, exist_ok=True)
    generate_site_table_basin(main_dir)

    # Fetch the Geonet data
    geonet.parse_geonet_information(
        main_dir,
        start_date,
        end_date,
        n_procs,
        geonet_batch_size,
        only_event_ids,
        only_sites,
    )

    # Merge the tectonic domains
    print("Merging tectonic domains")
    eq_source_ffp = flatfile_dir / "earthquake_source_table.csv"
    eq_tect_domain_ffp = flatfile_dir / "earthquake_source_table.csv"
    tect_domain.add_tect_domain(eq_source_ffp, eq_tect_domain_ffp, n_procs)

    # Generate the phase arrival table
    print("Generating phase arrival table")
    gen_phase_arrival_table.generate_phase_arrival_table(
        main_dir, flatfile_dir, n_procs
    )

    # Generate SNR
    print("Calculating SNR")
    snr_fas_output_dir = file_structure.get_snr_fas_dir(main_dir)
    phase_table_path = flatfile_dir / "phase_arrival_table.csv"
    calculate_snr(
        main_dir,
        phase_table_path,
        flatfile_dir,
        snr_fas_output_dir,
        n_procs,
        no_smoothing=no_smoothing,
        ko_matrix_path=ko_matrix_path,
        batch_size=snr_batch_size,
    )

    # Calculate Fmax
    print("Calculating Fmax")
    calc_fmax(main_dir, flatfile_dir, snr_fas_output_dir, n_procs)


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
    checkpoint: Annotated[
        bool,
        typer.Option(
            help="If True, the function will check for already completed files and skip them",
            is_flag=True,
        ),
    ] = False,
    n_procs: Annotated[int, typer.Option(help="The number of processes to use")] = 1,
):
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Run filtering and processing of mseeds
    gmc_ffp = flatfile_dir / "gmc_predictions.csv"
    fmax_ffp = flatfile_dir / "fmax.csv"
    process_records(main_dir, gmc_ffp, fmax_ffp, n_procs)

    # Run IM calculation
    im_dir = file_structure.get_im_dir(main_dir)
    run_im_calculation(main_dir, im_dir, n_procs, checkpoint)

    # Merge IM results
    merge_im_results(im_dir, flatfile_dir, gmc_ffp, fmax_ffp)

    # Calculate distances
    distances.calc_distances(main_dir, n_procs)

    # Merge flat files
    merge_flat_files(main_dir)


if __name__ == "__main__":
    app()
