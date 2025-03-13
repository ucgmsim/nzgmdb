"""
File that contains the function scripts that can be called to run the NZGMDB pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.calculation import distances, fmax, ims, snr
from nzgmdb.data_processing import merge_flatfiles, process_observed, quality_db
from nzgmdb.data_retrieval import geonet, sites, tect_domain
from nzgmdb.management import file_structure, shell_commands
from nzgmdb.phase_arrival import gen_phase_arrival_table
from nzgmdb.scripts import run_gmc, upload_to_dropbox

app = typer.Typer()


@app.command(
    help="Fetch earthquake data from Geonet and generates the earthquake source and station magnitude tables."
)
def fetch_geonet_data(  # noqa: D103
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
        list[str],
        typer.Option(
            help="A list of event ids to filter the earthquake data, separated by commas",
            callback=lambda x: [] if x is None else x[0].split(","),
        ),
    ] = None,
    only_sites: Annotated[
        list[str],
        typer.Option(
            help="A list of site names to filter the earthquake data, separated by commas",
            callback=lambda x: [] if x is None else x[0].split(","),
        ),
    ] = None,
    real_time: Annotated[
        bool,
        typer.Option(
            help="If True, the function will run in real time mode by using a different client",
        ),
    ] = False,
):
    geonet.parse_geonet_information(
        main_dir,
        start_date,
        end_date,
        n_procs,
        batch_size,
        only_event_ids,
        only_sites,
        real_time,
    )


@app.command(help="Add tectonic domains to the earthquake source table")
def merge_tect_domain(  # noqa: D103
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
def make_phase_arrival_table(  # noqa: D103
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
    run_phasenet_script_ffp: Annotated[
        Path,
        typer.Argument(
            help="The script full file path to run PhaseNet (In NZGMDB/phase_arrival).",
            exists=True,
            dir_okay=False,
        ),
    ],
    conda_sh: Annotated[
        Path,
        typer.Argument(
            help="Path to activate your mamba conda.sh script.",
        ),
    ],
    env_activate_command: Annotated[
        str,
        typer.Argument(
            help="The command to activate the environment for running PhaseNet.",
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
):
    gen_phase_arrival_table.generate_phase_arrival_table(
        main_dir,
        output_dir,
        run_phasenet_script_ffp,
        conda_sh,
        env_activate_command,
        n_procs,
    )


@app.command(
    help="Calculate the signal to noise ratio of the waveforms as well as FAS. "
    "Requires the phase arrival table and mseed files to be generated. "
    "Allows custom output directories for meta output directory, and SNR/FAS output. "
    "If not provided, the default directories are used as if running the full NZGMDB pipeline."
    "Note: Can't have the common frequency vector as an input due to typer limitations. "
    "Instead change the configuration file."
)
def calculate_snr(  # noqa: D103
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
            file_structure.get_flatfile_dir(main_dir)
            / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
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
        batch_size=batch_size,
    )


@app.command(
    help="Calculate the maximum useable frequency (fmax). "
    "Requires the snr_fas files and the snr metadata. "
    "Several parameters are set in the config file."
)
def calc_fmax(  # noqa: D103
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
    waveform_dir: Annotated[
        Path,
        typer.Option(
            help="Path to the directory containing the mseed files to process",
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
    if waveform_dir is None:
        waveform_dir = file_structure.get_waveform_dir(main_dir)
    if snr_fas_output_dir is None:
        snr_fas_output_dir = file_structure.get_snr_fas_dir(main_dir)

    fmax.run_full_fmax_calc(meta_output_dir, waveform_dir, snr_fas_output_dir, n_procs)


@app.command(
    help="Process the mseed files to txt files. "
    "Saves the skipped records to a csv file and gives reasons why they were skipped"
)
def process_records(  # noqa: D103
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
        gmc_ffp = (
            file_structure.get_flatfile_dir(main_dir)
            / file_structure.FlatfileNames.GMC_PREDICTIONS
        )
    if fmax_ffp is None:
        fmax_ffp = (
            file_structure.get_flatfile_dir(main_dir)
            / file_structure.FlatfileNames.FMAX
        )

    process_observed.process_mseeds_to_txt(main_dir, gmc_ffp, fmax_ffp, n_procs)


@app.command(help="Run IM Calculation on processed waveform files")
def run_im_calculation(  # noqa: D103
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
def generate_site_table_basin(  # noqa: D103
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

    site_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.SITE_TABLE, index=False
    )


@app.command(
    help="Calculate the distances between the earthquake source and the station"
)
def calculate_distances(  # noqa: D103
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
def merge_im_results(  # noqa: D103
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
def merge_flat_files(  # noqa: D103
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
    help="Create a quality database for the NZGMDB results by running quality checks"
)
def create_quality_db(  # noqa: D103
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (Highest level directory)",
            exists=True,
            file_okay=False,
        ),
    ],
    bypass_records_ffp: Annotated[
        Path,
        typer.Option(
            help="The full file path to the bypass records file",
            readable=True,
            exists=True,
            dir_okay=False,
        ),
    ] = None,
):
    quality_db.create_quality_db(main_dir, bypass_records_ffp)


@app.command(
    help="Run the Entire NZGMDB pipeline."
    "- Fetch Geonet data "
    "- Merge tectonic domains "
    "- Generate phase arrival table "
    "- Calculate SNR "
    "- Calculate Fmax"
    "- Run GMC"
    "- Process and filter waveform data to txt files "
    "- Calculate IM's "
    "- Merge IM results into flatfiles"
    "- Calculate distances"
    "- Merge flat files"
    "- Upload to Dropbox"
)
def run_full_nzgmdb(  # noqa: D103
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
    gm_classifier_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory for gm_classifier.",
        ),
    ],
    conda_sh: Annotated[
        Path,
        typer.Argument(
            help="Path to activate your mamba conda.sh script.",
        ),
    ],
    gmc_activate: Annotated[
        str,
        typer.Argument(
            help="Command to activate gmc environment for extracting features.",
        ),
    ],
    gmc_predict_activate: Annotated[
        str,
        typer.Argument(
            help="Command to activate gmc_predict environment to run the predictions.",
        ),
    ],
    gmc_procs: Annotated[
        int,
        typer.Option(
            help="Number of processes to use for GMC due to large memory requirement"
        ),
    ] = 1,
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
    checkpoint: Annotated[
        bool,
        typer.Option(
            help="If True, the function will check for already completed files and skip them",
        ),
    ] = False,
    only_event_ids: Annotated[
        list[str],
        typer.Option(
            help="A list of event ids to filter the earthquake data, separated by commas",
            callback=lambda x: [] if x is None else x[0].split(","),
        ),
    ] = None,
    only_sites: Annotated[
        list[str],
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
    real_time: Annotated[
        bool,
        typer.Option(
            help="If True, the function will run in real time mode by using a different client",
        ),
    ] = False,
    upload: Annotated[
        bool,
        typer.Option(
            help="If True, the function will upload the results to Dropbox",
        ),
    ] = False,
    create_quality_db: Annotated[
        bool,
        typer.Option(
            help="If True, the function will create a quality database",
        ),
    ] = False,
    bypass_records_ffp: Annotated[
        Path,
        typer.Option(
            help="The full file path to the bypass records file",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    custom_rrup: Annotated[
        float,
        typer.Option(
            help="Custom value for rrup to use for the distance calculation",
        ),
    ] = None,
):
    main_dir.mkdir(parents=True, exist_ok=True)

    # Generate the site basin flatfile
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    flatfile_dir.mkdir(parents=True, exist_ok=True)
    if not (
        checkpoint
        and (flatfile_dir / file_structure.PreFlatfileNames.SITE_TABLE).exists()
    ):
        print("Generating site table basin flatfile")
        generate_site_table_basin(main_dir)

    # Fetch the Geonet data
    if not (
        checkpoint
        and (
            flatfile_dir
            / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_GEONET
        ).exists()
    ):
        print("Fetching Geonet data")
        geonet.parse_geonet_information(
            main_dir,
            start_date,
            end_date,
            n_procs,
            geonet_batch_size,
            only_event_ids,
            only_sites,
            real_time,
            custom_rrup,
        )

    # Merge the tectonic domains
    if not (
        checkpoint
        and (
            flatfile_dir
            / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_TECTONIC
        ).exists()
    ):
        print("Merging tectonic domains")
        eq_source_ffp = (
            flatfile_dir
            / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_GEONET
        )
        eq_tect_domain_ffp = (
            flatfile_dir
            / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_TECTONIC
        )
        tect_domain.add_tect_domain(eq_source_ffp, eq_tect_domain_ffp, n_procs)

    # Generate the phase arrival table
    if not (
        checkpoint
        and (
            flatfile_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
        ).exists()
    ):
        print("Generating phase arrival table")
        run_phasenet_script_ffp = (
            Path(__file__).parent.parent / "phase_arrival/run_phasenet.py"
        )
        gen_phase_arrival_table.generate_phase_arrival_table(
            main_dir,
            flatfile_dir,
            run_phasenet_script_ffp,
            conda_sh,
            gmc_activate,
            n_procs,
        )

    # Generate SNR
    snr_fas_output_dir = file_structure.get_snr_fas_dir(main_dir)
    if not (
        checkpoint
        and (flatfile_dir / file_structure.FlatfileNames.SNR_METADATA).exists()
    ):
        print("Calculating SNR")
        phase_table_path = (
            flatfile_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
        )
        calculate_snr(
            main_dir,
            phase_table_path,
            flatfile_dir,
            snr_fas_output_dir,
            n_procs,
            batch_size=snr_batch_size,
        )

    # Calculate Fmax
    if not (checkpoint and (flatfile_dir / file_structure.FlatfileNames.FMAX).exists()):
        print("Calculating Fmax")
        waveform_dir = file_structure.get_waveform_dir(main_dir)
        calc_fmax(main_dir, flatfile_dir, waveform_dir, snr_fas_output_dir, n_procs)

    # Run GMC
    if not (
        checkpoint
        and (flatfile_dir / file_structure.FlatfileNames.GMC_PREDICTIONS).exists()
    ):
        print("Running GMC")
        run_gmc.run_gmc_processing(
            main_dir,
            gm_classifier_dir,
            ko_matrix_path,
            conda_sh,
            gmc_activate,
            gmc_predict_activate,
            gmc_procs,
        )

    # Run filtering and processing of mseeds
    gmc_ffp = flatfile_dir / file_structure.FlatfileNames.GMC_PREDICTIONS
    fmax_ffp = flatfile_dir / file_structure.FlatfileNames.FMAX
    if not (
        checkpoint
        and (
            flatfile_dir
            / file_structure.SkippedRecordFilenames.PROCESSING_SKIPPED_RECORDS
        ).exists()
    ):
        print("Processing records")
        process_records(main_dir, gmc_ffp, fmax_ffp, n_procs)

    # Run IM calculation
    im_dir = file_structure.get_im_dir(main_dir)
    im_dir.mkdir(parents=True, exist_ok=True)
    print("Calculating IMs")
    # Run IM Calc with a sub-command to manage single core issues
    im_calc_command = f"python {__file__} run-im-calculation {main_dir} --output-dir {im_dir} --n-procs {n_procs} {'--checkpoint' if checkpoint else ''}"
    env_activate_command = "conda activate NZGMDB_3_11"
    log_file_ffp = im_dir / "run_im_calculation.log"
    shell_commands.run_command(
        im_calc_command, conda_sh, env_activate_command, log_file_ffp
    )
    # run_im_calculation(main_dir, im_dir, n_procs, checkpoint)

    # Merge IM results
    if not (
        checkpoint
        and (
            flatfile_dir / file_structure.PreFlatfileNames.GROUND_MOTION_IM_CATALOGUE
        ).exists()
    ):
        print("Merging IM results")
        merge_im_results(im_dir, flatfile_dir, gmc_ffp, fmax_ffp)

    # Calculate distances
    if not (
        checkpoint
        and (
            flatfile_dir
            / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_DISTANCES
        ).exists()
    ):
        print("Calculating distances")
        distances.calc_distances(main_dir, n_procs)

    # Merge flat files
    if not (
        checkpoint
        and (
            flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD100_FLAT
        ).exists()
    ):
        print("Merging flat files")
        merge_flat_files(main_dir)

    if create_quality_db:
        print("Creating quality database")
        quality_db.create_quality_db(main_dir, bypass_records_ffp)

    # Upload to dropbox
    if upload:
        print("Uploading to Dropbox")
        upload_to_dropbox.upload_to_dropbox(main_dir, n_procs=n_procs)


if __name__ == "__main__":
    app()
