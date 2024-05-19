import functools
from pathlib import Path
import multiprocessing as mp
from typing import List, Dict

import obspy
import numpy as np
import pandas as pd

from qcore.constants import Components
from nzgmdb.mseed_management import reading
from nzgmdb.management import file_structure, config as cfg
from IM_calculation.IM import im_calculation
from IM_calculation.IM.read_waveform import Waveform


def compute_im_for_waveform(
    waveform: Waveform,
    record_id: str,
    event_output_path: Path,
    components: List[Components],
    ims: List[str],
    im_options: Dict[str, List[float]],
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
    components : List[Components]
        The components of the record
    ims : List[str]
        The IMs to calculate
    im_options : Dict[str, List[float]]
        The options for the IMs
    """
    im_result = im_calculation.compute_measure_single(
        (waveform, None),
        ims,
        components,
        im_options,
        components,
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
    components: List[Components],
    ims: List[str],
    im_options: Dict[str, List[float]],
):
    """
    Calculate the IMs for a single record and save the results to a csv file

    Parameters
    ----------
    ffp_000 : Path
        The full file path to the 000 component file
    output_path : Path
        The path to the output directory
    components : List[Components]
        The components of the record
    ims : List[str]
        The IMs to calculate
    im_options : Dict[str, List[float]]
        The options for the IMs
    """
    # Load the mseed file
    try:
        mseed_file = ffp_000.parent.parent / "mseed" / f"{ffp_000.stem}.mseed"
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
            "record_id": ffp_000.stem,
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
        waveform, ffp_000.stem, event_output_path, components, ims, im_options
    )

    return None


def compute_ims_for_all_processed_records(
    main_dir: Path,
    output_path: Path,
    n_procs: int = 1,
    checkpoint: bool = False,
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
    """
    # Get the waveform directory and all the 000 files
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    comp_000_files = waveform_dir.rglob("*.000")

    if checkpoint:
        # Get list of already completed files
        completed_files = [f.stem for f in output_path.glob("*_IM.csv")]
        # Remove completed files from the list
        comp_000_files = [f for f in comp_000_files if f.stem not in completed_files]

    # Set components from qcore class for IM calculation
    components = [Components.c090, Components.c000, Components.cver]

    # Load the config and extract the IM options
    config = cfg.Config()
    ims = config.get_value("ims")
    psa_periods = config.get_value("psa_periods")
    fas_frequencies = np.logspace(
        config.get_value("fas_start"),
        config.get_value("fas_end"),
        num=config.get_value("fas_num"),
        base=config.get_value("fas_base"),
    )

    im_options = {
        "pSA": psa_periods,
        "SDI": psa_periods,
        "FAS": im_calculation.validate_fas_frequency(fas_frequencies),
    }

    # Create the pool of processes
    with mp.Pool(n_procs) as pool:
        skipped_records = pool.map(
            functools.partial(
                calculate_im_for_record,
                output_path=output_path,
                components=components,
                ims=ims,
                im_options=im_options,
            ),
            comp_000_files,
        )

    # Concatenate all the skipped records
    skipped_records = pd.concat(skipped_records, ignore_index=True)

    # Save the skipped records
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    skipped_records.to_csv(flatfile_dir / "IM_calc_skipped_records.csv", index=False)
