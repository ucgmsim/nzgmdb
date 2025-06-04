"""
File that contains the function scripts that can be called to run the NZGMDB pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from IM.ims import IM
from nzgmdb.calculation import aftershocks, distances, fmax, ims, snr
from nzgmdb.data_processing import merge_flatfiles, process_observed, quality_db
from nzgmdb.data_retrieval import geonet, sites, tect_domain
from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from nzgmdb.phase_arrival import gen_phase_arrival_table
from nzgmdb.scripts import run_gmc, upload_to_dropbox
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


@cli.from_docstring(app)
def fetch_geonet_data(
    main_dir: Annotated[Path, typer.Argument(file_okay=False)],
    start_date: Annotated[datetime, typer.Argument()],
    end_date: Annotated[datetime, typer.Argument()],
    n_procs: Annotated[int, typer.Option()] = 1,
    batch_size: Annotated[int, typer.Option()] = 500,
    only_event_ids: Annotated[
        list[str], typer.Option(callback=lambda x: [] if x is None else x[0].split(","))
    ] = None,
    only_sites: Annotated[
        list[str], typer.Option(callback=lambda x: [] if x is None else x[0].split(","))
    ] = None,
    only_record_ids_ffp: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = None,
    real_time: Annotated[bool, typer.Option()] = False,
    mp_sites: Annotated[bool, typer.Option()] = False,
):
    """
    Fetch earthquake data from Geonet and generate the earthquake source and station magnitude tables.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (highest level directory).
    start_date : datetime
        The start date to filter the earthquake data.
    end_date : datetime
        The end date to filter the earthquake data.
    n_procs : int, optional
        Number of processes to use (default is 1).
    batch_size : int, optional
        The batch size for the Geonet data retrieval, specifying how many events to process at a time (default is 500).
    only_event_ids : list[str], optional
        A list of event IDs to filter the earthquake data, separated by commas (default is None).
    only_sites : list[str], optional
        A list of site names to filter the earthquake data, separated by commas (default is None).
    only_record_ids_ffp : Path, optional
        The full file path to a set of record IDs to only run for (default is None).
    real_time : bool, optional
        If True, the function will run in real-time mode by using a different client (default is False).
    mp_sites : bool, optional
        If True, the function will use multiprocessing over sites instead of events (default is False).
    """
    geonet.parse_geonet_information(
        main_dir,
        start_date,
        end_date,
        n_procs,
        batch_size,
        only_event_ids,
        only_sites,
        only_record_ids_ffp,
        real_time,
        mp_sites,
    )


@cli.from_docstring(app)
def merge_tect_domain(
    eq_source_ffp: Annotated[
        Path,
        typer.Argument(
            readable=True,
            exists=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
        ),
    ],
    n_procs: Annotated[int, typer.Option()] = 1,
):
    """
    Add tectonic domains to the earthquake source table.

    Parameters
    ----------
    eq_source_ffp : Path
        The file path to the earthquake source table.
    output_dir : Path
        The directory to save the earthquake source table with tectonic domains.
    n_procs : int, optional
        Number of processes to use (default is 1).
    """
    tect_domain.add_tect_domain(eq_source_ffp, output_dir, n_procs)


@cli.from_docstring(app)
def make_phase_arrival_table(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(file_okay=False),
    ],
    run_phasenet_script_ffp: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
        ),
    ],
    conda_sh: Annotated[
        Path,
        typer.Argument(),
    ],
    env_activate_command: Annotated[
        str,
        typer.Argument(),
    ],
    n_procs: Annotated[int, typer.Option()] = 1,
):
    """
    Generate the phase arrival table using mseed data and a P-wave picker.

    This function requires the mseed files to be generated beforehand.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
        Glob is used to find all mseed files recursively.
    output_dir : Path
        The directory to save the phase arrival table.
    run_phasenet_script_ffp : Path
        The script full file path to run PhaseNet (located in NZGMDB/phase_arrival).
    conda_sh : Path
        Path to activate your mamba conda.sh script.
    env_activate_command : str
        The command to activate the environment for running PhaseNet.
    n_procs : int, optional
        Number of processes to use (default is 1).
    """
    gen_phase_arrival_table.generate_phase_arrival_table(
        main_dir,
        output_dir,
        run_phasenet_script_ffp,
        conda_sh,
        env_activate_command,
        n_procs,
    )


@cli.from_docstring(app)
def calculate_snr(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    ko_directory: Annotated[
        Path,
        typer.Argument(
            help="The directory containing the Konno-Ohmachi smoothing files",
            exists=True,
            file_okay=False,
        ),
    ],
    phase_table_path: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
        ),
    ] = None,
    meta_output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
        ),
    ] = None,
    snr_fas_output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
        ),
    ] = None,
    n_procs: Annotated[int, typer.Option()] = 1,
    batch_size: Annotated[
        int,
        typer.Option(),
    ] = 5000,
    bypass_records_ffp: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Calculate the signal-to-noise ratio (SNR) of waveforms and compute the Fourier Amplitude Spectrum (FAS).

    This function requires the phase arrival table and mseed files to be generated beforehand.
    Allows custom output directories for metadata, SNR, and FAS output. If not provided,
    the default directories are used as if running the full NZGMDB pipeline.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    ko_directory : Path
        The directory containing the Konno-Ohmachi smoothing files.
    phase_table_path : Path, optional
        Path to the phase arrival table. If not provided, defaults to the expected location.
    meta_output_dir : Path, optional
        Path to the output directory for metadata and skipped records. Defaults to the expected location.
    snr_fas_output_dir : Path, optional
        Path to the output directory for the SNR and FAS data. Defaults to the expected location.
    n_procs : int, optional
        Number of processes to use (default is 1).
    batch_size : int, optional
        The batch size for SNR calculation (default is 5000).
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file for custom P-wave index values.
    """
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
        ko_directory,
        n_procs,
        batch_size=batch_size,
        bypass_records_ffp=bypass_records_ffp,
    )


@cli.from_docstring(app)
def calc_fmax(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    meta_output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
        ),
    ] = None,
    waveform_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
        ),
    ] = None,
    snr_fas_output_dir: Annotated[
        Path,
        typer.Option(
            file_okay=False,
        ),
    ] = None,
    n_procs: Annotated[int, typer.Option()] = 1,
    bypass_records_ffp: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Calculate the maximum usable frequency (fmax) for waveforms.

    This function requires the SNR/FAS files and SNR metadata. Several parameters
    are configured in the config file.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    meta_output_dir : Path, optional
        Path to the output directory for metadata and skipped records. Defaults to the expected location.
    waveform_dir : Path, optional
        Path to the directory containing the mseed files to process. Defaults to the expected location.
    snr_fas_output_dir : Path, optional
        Path to the output directory for the SNR and FAS data. Defaults to the expected location.
    n_procs : int, optional
        Number of processes to use (default is 1).
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file for custom fmax values.
    """
    if meta_output_dir is None:
        meta_output_dir = file_structure.get_flatfile_dir(main_dir)
    if waveform_dir is None:
        waveform_dir = file_structure.get_waveform_dir(main_dir)
    if snr_fas_output_dir is None:
        snr_fas_output_dir = file_structure.get_snr_fas_dir(main_dir)

    fmax.run_full_fmax_calc(
        meta_output_dir, waveform_dir, snr_fas_output_dir, n_procs, bypass_records_ffp
    )


@cli.from_docstring(app)
def process_records(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    gmc_ffp: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
        ),
    ] = None,
    fmax_ffp: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
        ),
    ] = None,
    bypass_records_ffp: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    n_procs: Annotated[int, typer.Option()] = 1,
):
    """
    Process mseed files into txt files and log skipped records.

    This function converts mseed files to txt format and saves skipped records
    with reasons in a CSV file.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    gmc_ffp : Path, optional
        The full file path to the GMC predictions file. Defaults to the expected location.
    fmax_ffp : Path, optional
        The full file path to the Fmax file. Defaults to the expected location.
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file.
    n_procs : int, optional
        The number of processes to use (default is 1).
    """
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

    process_observed.process_mseeds_to_txt(
        main_dir, gmc_ffp, fmax_ffp, bypass_records_ffp, n_procs
    )


@cli.from_docstring(app)
def run_im_calculation(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    ko_directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(file_okay=False),
    ] = None,
    n_procs: Annotated[int, typer.Option()] = 1,
    checkpoint: Annotated[
        bool,
        typer.Option(
            is_flag=True,
        ),
    ] = False,
    intensity_measures: Annotated[
        list[IM],
        typer.Option(
            callback=lambda x: [IM(i) for i in x[0].split(",")],
        ),
    ] = None,
):
    """
    Run IM Calculation on processed waveform files.

    This function computes intensity measures (IMs) for processed waveform files.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    ko_directory : Path
        The directory containing the Konno-Ohmachi smoothing files. Defaults to the expected location.
    output_dir : Path, optional
        The directory to save the IM files. Defaults to the expected location.
    n_procs : int, optional
        The number of processes to use (default is 1).
    checkpoint : bool, optional
        If True, the function will check for already completed files and skip them.
    intensity_measures : list[IM], optional
        The list of intensity measures to calculate, by default None and will use the config file.
    """
    if output_dir is None:
        output_dir = file_structure.get_im_dir(main_dir)
    ims.compute_ims_for_all_processed_records(
        main_dir, output_dir, ko_directory, n_procs, checkpoint, intensity_measures
    )


@cli.from_docstring(app)
def generate_site_table_basin(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
):
    """
    Generate the site table basin flatfile.

    This function creates a site table with basin information and saves it as a flatfile.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    """
    main_dir.mkdir(parents=True, exist_ok=True)
    # Generate the site basin flatfile
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    flatfile_dir.mkdir(parents=True, exist_ok=True)

    site_df = sites.create_site_table_response()
    site_df = sites.add_site_basins(site_df)

    site_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.SITE_TABLE, index=False
    )


@cli.from_docstring(app)
def calculate_distances(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="The number of processes to use")] = 1,
):
    """
    Calculate the distances between the earthquake source and the station.

    This function computes the distances between earthquake sources and seismic stations
    and saves the results to the appropriate output location.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    n_procs : int, optional
        The number of processes to use, by default 1.
    """
    distances.calc_distances(main_dir, n_procs)


@cli.from_docstring(app)
def calculate_aftershocks(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
):
    """
    Calculate the aftershock flags for the earthquake source table.

    This function determines whether earthquakes in the source table are classified as aftershocks
    based on predefined criteria and updates the table accordingly.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    """
    aftershocks.merge_aftershocks(main_dir)


@cli.from_docstring(app)
def merge_im_results(
    im_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(file_okay=False),
    ],
    gmc_ffp: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
        ),
    ] = None,
    fmax_ffp: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
        ),
    ] = None,
):
    """
    Merge IM results together into one flatfile and perform a filter for Ds595.

    This function consolidates individual IM result files into a single comprehensive
    dataset, ensuring consistency and filtering for the Ds595 parameter.

    Parameters
    ----------
    im_dir : Path
        The directory containing the IM results to merge.
    output_dir : Path
        The directory to save the merged IM file.
    gmc_ffp : Path
        The full file path to the GMC predictions file.
    fmax_ffp : Path
        The full file path to the Fmax file.
    """
    merge_flatfiles.merge_im_data(im_dir, output_dir, gmc_ffp, fmax_ffp)


@cli.from_docstring(app)
def merge_flat_files(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    bypass_records_ffp: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Merge all flatfiles together for final output and ensure correct filtering for only results with IM values.

    This function consolidates various flatfiles into a single output file while ensuring that only results
    containing IM values are included. It also integrates bypass records if provided.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file, if applicable.
    """
    merge_flatfiles.merge_flatfiles(main_dir, bypass_records_ffp)


@cli.from_docstring(app)
def create_quality_db(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    bypass_records_ffp: Annotated[
        Path,
        typer.Option(
            readable=True,
            exists=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Create a quality database for the NZGMDB results by running quality checks.

    This function generates a quality database by performing various quality checks on the NZGMDB results.
    It ensures that data integrity and consistency are maintained across the dataset.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file, if applicable.
    """
    quality_db.create_quality_db(main_dir, bypass_records_ffp)


@cli.from_docstring(app)
def run_full_nzgmdb(
    main_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    start_date: Annotated[
        datetime,
        typer.Argument(),
    ],
    end_date: Annotated[
        datetime,
        typer.Argument(),
    ],
    gm_classifier_dir: Annotated[
        Path,
        typer.Argument(),
    ],
    conda_sh: Annotated[
        Path,
        typer.Argument(),
    ],
    gmc_activate: Annotated[
        str,
        typer.Argument(),
    ],
    gmc_predict_activate: Annotated[
        str,
        typer.Argument(),
    ],
    ko_matrix_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    gmc_procs: Annotated[
        int,
        typer.Option(),
    ] = 1,
    n_procs: Annotated[int, typer.Option()] = 1,
    checkpoint: Annotated[
        bool,
        typer.Option(),
    ] = False,
    only_event_ids: Annotated[
        list[str],
        typer.Option(
            callback=lambda x: [] if x is None else x[0].split(","),
        ),
    ] = None,
    only_sites: Annotated[
        list[str],
        typer.Option(
            callback=lambda x: [] if x is None else x[0].split(","),
        ),
    ] = None,
    only_record_ids_ffp: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    geonet_batch_size: Annotated[
        int,
        typer.Option(),
    ] = 500,
    snr_batch_size: Annotated[
        int,
        typer.Option(),
    ] = 5000,
    real_time: Annotated[
        bool,
        typer.Option(),
    ] = False,
    upload: Annotated[
        bool,
        typer.Option(),
    ] = False,
    create_quality_db: Annotated[
        bool,
        typer.Option(),
    ] = False,
    bypass_records_ffp: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    machine: Annotated[
        cfg.MachineName | None,
        typer.Option(
            case_sensitive=False,
        ),
    ] = None,
):
    """
    Run the Entire NZGMDB pipeline.

    This function orchestrates the full pipeline of NZGMDB, executing all necessary steps sequentially.

    Steps Included:
    - Fetch Geonet data
    - Merge tectonic domains
    - Generate phase arrival table
    - Calculate SNR
    - Calculate Fmax
    - Run GMC
    - Process and filter waveform data to txt files
    - Calculate IMs
    - Merge IM results into flatfiles
    - Calculate distances
    - Merge flat files
    - Upload results to Dropbox (if specified)

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    start_date : datetime
        The start date to filter the earthquake data.
    end_date : datetime
        The end date to filter the earthquake data.
    gm_classifier_dir : Path
        Directory for gm_classifier.
    conda_sh : Path
        Path to activate your mamba conda.sh script.
    gmc_activate : str
        Command to activate gmc environment for extracting features.
    gmc_predict_activate : str
        Command to activate gmc_predict environment to run the predictions.
    ko_matrix_path : Path
        Path to the ko matrix directory
    gmc_procs : int, optional
        Number of processes to use for GMC (default is 1).
    n_procs : int, optional
        The number of processes to use (default is 1).
    checkpoint : bool, optional
        If True, the function will check for already completed files and skip them (default is False).
    only_event_ids : list[str], optional
        A list of event ids to filter the earthquake data, separated by commas.
    only_sites : list[str], optional
        A list of site names to filter the earthquake data, separated by commas.
    only_record_ids_ffp : Path, optional
        The full file path to a set of record_ids to process only those records.
    geonet_batch_size : int, optional
        The batch size for Geonet data retrieval (default is 500).
    snr_batch_size : int, optional
        The batch size for the SNR calculation (default is 5000).
    real_time : bool, optional
        If True, the function will run in real-time mode using a different client (default is False).
    upload : bool, optional
        If True, the function will upload the results to Dropbox (default is False).
    create_quality_db : bool, optional
        If True, the function will create a quality database (default is False).
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file, if applicable.
    machine : cfg.MachineName, optional
        The machine name to use for process configuration (default is None).
    """
    main_dir.mkdir(parents=True, exist_ok=True)
    config = cfg.Config()

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
        geo_n_procs = (
            n_procs
            if machine is None
            else config.get_n_procs(machine, cfg.WorkflowStep.GEONET)
        )
        geonet.parse_geonet_information(
            main_dir,
            start_date,
            end_date,
            geo_n_procs,
            geonet_batch_size,
            only_event_ids,
            only_sites,
            only_record_ids_ffp,
            real_time,
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
        tect_n_procs = (
            n_procs
            if machine is None
            else config.get_n_procs(machine, cfg.WorkflowStep.TEC_DOMAIN)
        )
        tect_domain.add_tect_domain(eq_source_ffp, eq_tect_domain_ffp, tect_n_procs)

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
        phase_n_procs = (
            n_procs
            if machine is None
            else config.get_n_procs(machine, cfg.WorkflowStep.PHASE_TABLE)
        )
        gen_phase_arrival_table.generate_phase_arrival_table(
            main_dir,
            flatfile_dir,
            run_phasenet_script_ffp,
            conda_sh,
            gmc_activate,
            phase_n_procs,
            bypass_records_ffp,
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
        snr_n_procs = (
            n_procs
            if machine is None
            else config.get_n_procs(machine, cfg.WorkflowStep.SNR)
        )
        calculate_snr(
            main_dir,
            ko_matrix_path,
            phase_table_path,
            flatfile_dir,
            snr_fas_output_dir,
            snr_n_procs,
            batch_size=snr_batch_size,
            bypass_records_ffp=bypass_records_ffp,
        )

    # Calculate Fmax
    if not (checkpoint and (flatfile_dir / file_structure.FlatfileNames.FMAX).exists()):
        print("Calculating Fmax")
        waveform_dir = file_structure.get_waveform_dir(main_dir)
        fmax_n_procs = (
            n_procs
            if machine is None
            else config.get_n_procs(machine, cfg.WorkflowStep.FMAX)
        )
        calc_fmax(
            main_dir,
            flatfile_dir,
            waveform_dir,
            snr_fas_output_dir,
            fmax_n_procs,
            bypass_records_ffp,
        )

    # Run GMC
    if not (
        checkpoint
        and (flatfile_dir / file_structure.FlatfileNames.GMC_PREDICTIONS).exists()
    ):
        print("Running GMC")
        gmc_n_procs = (
            gmc_procs
            if machine is None
            else config.get_n_procs(machine, cfg.WorkflowStep.GMC)
        )
        run_gmc.run_gmc_processing(
            main_dir,
            gm_classifier_dir,
            ko_matrix_path,
            conda_sh,
            gmc_activate,
            gmc_predict_activate,
            gmc_n_procs,
            bypass_records_ffp=bypass_records_ffp,
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
        process_n_procs = (
            n_procs
            if machine is None
            else config.get_n_procs(machine, cfg.WorkflowStep.PROCESS)
        )
        process_records(
            main_dir, gmc_ffp, fmax_ffp, bypass_records_ffp, process_n_procs
        )

    # Run IM calculation
    im_dir = file_structure.get_im_dir(main_dir)
    im_dir.mkdir(parents=True, exist_ok=True)
    print("Calculating IMs")
    im_n_procs = (
        n_procs if machine is None else config.get_n_procs(machine, cfg.WorkflowStep.IM)
    )
    run_im_calculation(main_dir, im_dir, ko_matrix_path, im_n_procs, checkpoint)

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
        dist_n_procs = (
            n_procs
            if machine is None
            else config.get_n_procs(machine, cfg.WorkflowStep.DISTANCES)
        )
        distances.calc_distances(main_dir, dist_n_procs)

    # Calculate aftershocks
    if not (
        checkpoint
        and (
            flatfile_dir
            / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_AFTERSHOCKS
        ).exists()
    ):
        print("Calculating aftershocks")
        aftershocks.merge_aftershocks(main_dir)

    # Merge flat files
    if not (
        checkpoint
        and (
            flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD100_FLAT
        ).exists()
    ):
        print("Merging flat files")
        merge_flat_files(main_dir, bypass_records_ffp)

    if create_quality_db:
        print("Creating quality database")
        quality_db.create_quality_db(main_dir, bypass_records_ffp)

    # Upload to dropbox
    if upload:
        print("Uploading to Dropbox")
        up_n_procs = (
            n_procs
            if machine is None
            else config.get_n_procs(machine, cfg.WorkflowStep.UPLOAD)
        )
        upload_to_dropbox.upload_to_dropbox(main_dir, n_procs=up_n_procs)


@cli.from_docstring(app)
def merge_databases(
    flatfile_db_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    to_merge_db_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    output_ffp: Annotated[
        Path,
        typer.Argument(),
    ],
):
    """
    Merge two databases together, allowing one to overwrite the other if duplicates are found.

    Parameters
    ----------
    flatfile_db_dir : Path
        The flatfile directory of the NZGMDB results (where the flatfiles are located).
    to_merge_db_dir : Path
        The flatfile directory of the NZGMDB results to replace and add to the main database.
    output_ffp : Path
        The full file path to place the output flatfiles for the merged database.
    """
    merge_flatfiles.merge_dbs(flatfile_db_dir, to_merge_db_dir, output_ffp)


if __name__ == "__main__":
    app()
