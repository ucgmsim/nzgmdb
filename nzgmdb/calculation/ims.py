import multiprocessing as mp
import queue
import time
from pathlib import Path

import numpy as np
import obspy
import pandas as pd

from IM_calculation.IM import im_calculation
from IM_calculation.IM.read_waveform import Waveform
from nzgmdb.management import config as cfg
from nzgmdb.management import custom_multiprocess, file_structure
from nzgmdb.mseed_management import reading
from qcore.constants import Components


def compute_im_for_waveform(
    waveform: Waveform,
    record_id: str,
    event_output_path: Path,
    components: list[Components],
    ims: list[str],
    im_options: dict[str, list[float]],
    ko_matrices_path: Path = None,
):
    """
    Compute the IMs for a single waveform and save the results to a csv file

    Parameters
    ----------
    waveform : Waveform
        The waveform object to calculate the IMs for
    record_id : str
        The record id
    event_output_path : Path
        The path to the event output directory
    components : list[Components]
        The components of the record
    ims : list[str]
        The IMs to calculate
    im_options : dict[str, list[float]]
        The options for the IMs
    ko_matrices_path : Path, optional
        The path to the KO matrices, by default None
    """
    im_result = im_calculation.compute_measure_single(
        (waveform, None),
        ims,
        components,
        im_options,
        components,
        ko_matrices_path=ko_matrices_path,
    )

    # Turn the results into a dataframe
    im_result_df = pd.DataFrame(im_result).T

    # Set a column for the mseed stem and then component and set at the front
    im_result_df.insert(0, "component", [comp.str_value for comp in components])
    im_result_df.insert(0, "record_id", record_id)

    # Save the file
    im_result_df.to_csv(event_output_path / f"{record_id}_IM.csv", index=False)


def calculate_im_for_record(
    ffp_000: Path,
    output_path: Path,
    components: list[Components],
    ims: list[str],
    im_options: dict[str, list[float]],
    ko_matrices_path: Path = None,
):
    """
    Calculate the IMs for a single record and save the results to a csv file

    Parameters
    ----------
    ffp_000 : Path
        The full file path to the 000 component file
    output_path : Path
        The path to the output directory
    components : list[Components]
        The components of the record
    ims : list[str]
        The IMs to calculate
    im_options : dict[str, list[float]]
        The options for the IMs
    ko_matrices_path : Path, optional
        The path to the KO matrices, by default None
    """
    # Load the mseed file
    mseed_file = ffp_000.parent.parent / "mseed" / f"{ffp_000.stem}.mseed"
    try:
        mseed = obspy.read(mseed_file)
    except FileNotFoundError:
        skipped_record_dict = {
            "record_id": ffp_000.stem,
            "reason": "Failed to find the mseed file",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Get the 090 and ver components full file paths
    ffp_090 = ffp_000.parent / f"{ffp_000.stem}.090"
    ffp_ver = ffp_000.parent / f"{ffp_000.stem}.ver"

    try:
        waveform = reading.create_waveform_from_processed(
            ffp_000, ffp_090, ffp_ver, delta=mseed[0].stats.delta
        )
    except FileNotFoundError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "Failed to find all components",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Get the event_id and create the output directory
    event_id = file_structure.get_event_id_from_mseed(mseed_file)
    event_output_path = output_path / event_id
    event_output_path.mkdir(exist_ok=True, parents=True)

    # Calculate the IMs
    compute_im_for_waveform(
        waveform,
        ffp_000.stem,
        event_output_path,
        components,
        ims,
        im_options,
        ko_matrices_path,
    )


def compute_ims_for_all_processed_records(
    main_dir: Path,
    output_path: Path,
    n_procs: int = 1,
    checkpoint: bool = False,
    ko_matrices_path: Path = None,
):
    """
    Compute the IMs for all processed records in the main directory

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    output_path : Path
        The path to the output directory
    n_procs : int, optional
        The number of processes to use
    checkpoint : bool, optional
        If True, the function will check for already completed files and skip them
    ko_matrices_path : Path, optional
        The path to the KO matrices, by default None
    """
    # Get the waveform directory and all the 000 files
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    comp_000_files = waveform_dir.rglob("*.000")

    if checkpoint:
        # Get list of already completed files and remove _IM suffix
        completed_files = [f.stem[:-3] for f in output_path.rglob("*_IM.csv")]
        # Remove completed files from the list
        comp_000_files = [f for f in comp_000_files if f.stem not in completed_files]

    print(f"Calculating IMs for {len(comp_000_files)} records")

    # Load the config and extract the IM options
    config = cfg.Config()
    ims = config.get_value("ims")
    psa_periods = np.asarray(config.get_value("psa_periods"))
    fas_frequencies = np.logspace(
        np.log10(config.get_value("common_frequency_start")),
        np.log10(config.get_value("common_frequency_end")),
        num=config.get_value("common_frequency_num"),
    )
    # Set components from qcore class for IM calculation
    _, components = Components.get_comps_to_calc_and_store(
        config.get_value("components")
    )

    im_options = {
        "pSA": psa_periods,
        "SDI": psa_periods,
        "FAS": im_calculation.validate_fas_frequency(fas_frequencies),
    }

    # Use custom_multiprocess to process the records
    skipped_records = custom_multiprocess.custom_multiprocess(
        calculate_im_for_record,
        comp_000_files,
        n_procs,
        output_path,
        components,
        ims,
        im_options,
        ko_matrices_path,
    )

    print("Finished calculating IMs")

    # Save the skipped records
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Check that there are skipped_records dataframes that are not None
    if not all(value is None for value in skipped_records):
        skipped_records_df = pd.concat(skipped_records).reset_index(drop=True)
    else:
        print("No skipped records")
        skipped_records_df = pd.DataFrame(columns=["record_id", "reason"])

    if checkpoint:
        # Add the skipped records to the existing skipped records
        try:
            existing_skipped_records = pd.read_csv(
                flatfile_dir
                / file_structure.SkippedRecordFilenames.IM_CALC_SKIPPED_RECORDS
            )
            skipped_records_df = pd.concat(
                [existing_skipped_records, skipped_records_df]
            ).reset_index(drop=True)
        except FileNotFoundError:
            pass

    skipped_records_df.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.IM_CALC_SKIPPED_RECORDS,
        index=False,
    )
