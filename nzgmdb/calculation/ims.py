"""
This module contains functions for calculating the Intensity Measures (IMs) for the NZGMDB records.
"""

import functools
import multiprocessing as mp


# if __name__ == "__main__":


import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from IM import im_calculation, ims, waveform_reading
from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
import cProfile
import pstats
import io


def calculate_im_for_record_profiled(*args, **kwargs):
    # profiler = cProfile.Profile()
    # profiler.enable()

    result = calculate_im_for_record(*args, **kwargs)  # Your original function

    # profiler.disable()
    # s = io.StringIO()
    # stats = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(15)  # Print top 10 slowest functions
    # print(s.getvalue())  # Print profiling results to stdout

    return result


def calculate_im_for_record(
    ffp_000: Path,
    output_path: Path,
    intensity_measures: list[ims.IM],
    psa_periods: np.ndarray,
    fas_frequencies: np.ndarray,
    ko_directory: Path,
):
    """
    Calculate the IMs for a single record and save the results to a csv file

    Parameters
    ----------
    ffp_000 : Path
        The full file path to the 000 component file
    output_path : Path
        The path to the output directory
    intensity_measures : list[ims.IM]
        The list of intensity measures to calculate
    psa_periods : np.ndarray
        The periods for calculating the pseudo-spectral acceleration
    fas_frequencies : np.ndarray
        The frequencies for calculating the Fourier amplitude spectrum
    ko_bandwith : int, optional
        The bandwidth for the Konno-Ohmachi smoothing, by default 40

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the record_id and the reason for skipping the record only if the record was skipped
    """
    # Get the 090 and ver components full file paths
    ffp_090 = ffp_000.parent / f"{ffp_000.stem}.090"
    ffp_ver = ffp_000.parent / f"{ffp_000.stem}.ver"
    record_id = ffp_000.stem

    try:
        dt, waveform = waveform_reading.read_ascii(ffp_000, ffp_090, ffp_ver)
    except FileNotFoundError:
        skipped_record_dict = {
            "record_id": record_id,
            "reason": "Failed to find the waveform ascii files",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    print(f"Calculating IMs for {record_id}")

    # Get the event_id and create the output directory
    event_id = file_structure.get_event_id_from_mseed(ffp_000)
    event_output_path = output_path / event_id
    event_output_path.mkdir(exist_ok=True, parents=True)

    nyquist_feq = (1 / dt) * 0.5
    # Filter the fas_frequencies to be less than or equal to the nyquist frequency
    fas_frequencies = fas_frequencies[fas_frequencies <= nyquist_feq]

    # Calculate the IMs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        im_result_df = im_calculation.calculate_ims(
            waveform,
            dt,
            intensity_measures,
            psa_periods,
            fas_frequencies,
            cores=1,
            ko_directory=ko_directory,
        )

    print(f"Saving IMs for {record_id}")

    # Set a column for the record_id and then component and set at the front
    im_result_df = im_result_df.reset_index()
    im_result_df = im_result_df.rename(columns={"index": "component"})
    im_result_df.insert(0, "record_id", record_id)

    # Save the file
    im_result_df.to_csv(event_output_path / f"{record_id}_IM.csv", index=False)


def compute_ims_for_all_processed_records(
    main_dir: Path,
    output_path: Path,
    ko_directory: Path,
    n_procs: int = 1,
    checkpoint: bool = False,
    intensity_measures: list[ims.IM] = None,
):
    """
    Compute the IMs for all processed records in the main directory

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    output_path : Path
        The path to the output directory
    ko_directory : Path
        The path to the directory containing the Konno-Ohmachi smoothing files
    n_procs : int, optional
        The number of processes to use
    checkpoint : bool, optional
        If True, the function will check for already completed files and skip them
    intensity_measures : list[ims.IM], optional
        The list of intensity measures to calculate, by default None and will use the config file
    """
    # Get the waveform directory and all the 000 files
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    comp_000_files = list(waveform_dir.rglob("*.000"))

    if checkpoint:
        # Get list of already completed files and remove _IM suffix
        completed_files = [f.stem[:-3] for f in output_path.rglob("*_IM.csv")]
        # Remove completed files from the list
        comp_000_files = [f for f in comp_000_files if f.stem not in completed_files]

    print(f"Calculating IMs for {len(comp_000_files)} records")

    # Load the config and extract the IM options
    config = cfg.Config()
    intensity_measures = (
        [ims.IM[measure] for measure in config.get_value("ims")]
        if intensity_measures is None
        else intensity_measures
    )
    psa_periods = np.asarray(config.get_value("psa_periods"))
    fas_frequencies = np.logspace(
        np.log10(config.get_value("common_frequency_start")),
        np.log10(config.get_value("common_frequency_end")),
        num=config.get_value("common_frequency_num"),
    )

    # This is a fix for multiprocessing issues in IM calculation
    mp.set_start_method("spawn", force=True)

    # Fetch results
    with mp.Pool(n_procs) as p:
        skipped_records = p.map(
            functools.partial(
                calculate_im_for_record,
                output_path=output_path,
                intensity_measures=intensity_measures,
                psa_periods=psa_periods,
                fas_frequencies=fas_frequencies,
                ko_directory=ko_directory,
            ),
            comp_000_files,
        )

    mp.set_start_method("fork", force=True)

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
