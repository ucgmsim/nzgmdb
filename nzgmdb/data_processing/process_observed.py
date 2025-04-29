"""
This module contains functions to process observed data from mseed files and turn them into ascii files
"""

import functools
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

import qcore.timeseries as ts
from nzgmdb.data_processing import waveform_manipulation
from nzgmdb.management import custom_errors, file_structure
from nzgmdb.mseed_management import reading


def process_single_mseed(
    mseed_file: Path,
    gmc_df: pd.DataFrame,
    fmax_df: pd.DataFrame,
    bypass_df: pd.DataFrame = None,
):
    """
    Process a single mseed file and save the processed data to a txt file
    Will return a dataframe containing the skipped record name and reason why
    if the record must be skipped due to either not containing 3 components,
    failing to find the inventory information or the lowcut frequency being
    greater than the highcut frequency during processing

    Parameters
    ----------
    mseed_file : Path
        The path to the mseed file
    gmc_df : pd.DataFrame
        The GMC values containing fmin information
    fmax_df : pd.DataFrame
        The Fmax values
    bypass_df : pd.DataFrame
        The bypass records containing custom fmin, fmax values

    Returns
    -------
    pd.DataFrame | None
        Dataframe containing the skipped record name and reason why
        or None if the record was processed successfully
    """
    # Check if the mseed file is in the GMC predictions
    mseed_stem = mseed_file.stem
    gmc_rows = gmc_df[gmc_df["record"] == mseed_stem]

    # Read mseed information
    mseed = reading.read_mseed_to_stream(mseed_file)

    # Extract mseed values
    dt = mseed.traces[0].stats.delta
    station = mseed.traces[0].stats.station

    # Check the length of the mseed file for 3 components
    if len(mseed) != 3:
        skipped_record_dict = {
            "record_id": mseed_stem,
            "reason": "File did not contain 3 components",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Perform initial pre-processing
    try:
        mseed = waveform_manipulation.initial_preprocessing(mseed)
    except custom_errors.InventoryNotFoundError:
        skipped_record_dict = {
            "record_id": mseed_stem,
            "reason": "Failed to find Inventory information",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record
    except custom_errors.SensitivityRemovalError:
        skipped_record_dict = {
            "record_id": mseed_stem,
            "reason": "Failed to remove sensitivity",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record
    except custom_errors.RotationError:
        skipped_record_dict = {
            "record_id": mseed_stem,
            "reason": "Failed to rotate the data",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Get the GMC fmin and fmax values
    fmin = None if gmc_rows.empty else gmc_rows["fmin_mean"].max()
    fmax_rows = fmax_df[fmax_df["record_id"] == mseed_stem]
    fmax = (
        None
        if fmax_rows.empty
        else min(fmax_rows.loc[:, ["fmax_000", "fmax_090", "fmax_ver"]].values[0])
    )

    # Check if the record is in the bypass records (Only if there wasnt an existing fmin, fmax)
    if bypass_df is not None and fmin is None or fmax is None:
        if mseed_stem in bypass_df["record_id"].values:
            bypass_row = bypass_df[bypass_df["record_id"] == mseed_stem]
            fmin_bypass = max(
                bypass_row.loc[:, ["fmin_000", "fmin_090", "fmin_ver"]].values[0]
            )
            fmax_bypass = min(
                bypass_row.loc[:, ["fmax_000", "fmax_090", "fmax_ver"]].values[0]
            )
            fmin = None if np.isnan(fmin_bypass) else fmin_bypass
            fmax = None if np.isnan(fmax_bypass) else fmax_bypass

    # Perform high and lowcut processing
    try:
        (
            acc_bb_000,
            acc_bb_090,
            acc_bb_ver,
        ) = waveform_manipulation.high_and_low_cut_processing(mseed, dt, fmin, fmax)
    except custom_errors.InvalidTraceLengthError:
        skipped_record_dict = {
            "record_id": mseed_stem,
            "reason": "Invalid trace length for the mseed file",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record
    except custom_errors.LowcutHighcutError:
        skipped_record_dict = {
            "record_id": mseed_stem,
            "reason": "Lowcut frequency is greater than the highcut frequency",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record
    except custom_errors.ComponentSelectionError:
        skipped_record_dict = {
            "record_id": mseed_stem,
            "reason": "Failed to find N, E, X, or Y components",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record
    except custom_errors.DigitalFilterError:
        skipped_record_dict = {
            "record_id": mseed_stem,
            "reason": "Failed to apply bandpass filter",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Create the output directory
    output_dir = file_structure.get_processed_dir_from_mseed(mseed_file)
    output_dir.mkdir(exist_ok=True)

    # Write the data to the output directory
    for comp, acc_bb in zip(
        ["000", "090", "ver"], [acc_bb_000, acc_bb_090, acc_bb_ver]
    ):
        filename = output_dir / f"{mseed_stem}.{comp}"
        ts.timeseries_to_text(
            acc_bb,
            filename,
            dt,
            station,
            comp,
        )


def process_mseeds_to_txt(
    main_dir: Path,
    gmc_ffp: Path,
    fmax_ffp: Path,
    bypass_records_ffp: Path = None,
    n_procs: int = 1,
):
    """
    Process the mseed files to txt files
    Saves the skipped records to a csv file and gives reasons why they were skipped

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    gmc_ffp : Path
        The full file path to the GMC predictions file
    fmax_ffp : Path
        The full file path to the Fmax file
    bypass_records_ffp : Path
        The full file path to the bypass records file, which includes a custom fmin, fmax
    n_procs : int
        The number of processes to use for multiprocessing
    """
    # Get the raw waveform mseed files
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    mseed_files = waveform_dir.rglob("*.mseed")

    # Load the GMC, Fmax and bypass records
    gmc_df = pd.read_csv(gmc_ffp)
    try:
        fmax_df = pd.read_csv(fmax_ffp)
    except pd.errors.EmptyDataError:
        fmax_df = pd.DataFrame(
            columns=["record_id", "fmax_000", "fmax_090", "fmax_ver"]
        )
    bypass_df = None if bypass_records_ffp is None else pd.read_csv(bypass_records_ffp)

    # Use multiprocessing to process the mseed files
    with multiprocessing.Pool(processes=n_procs) as pool:
        skipped_records = pool.map(
            functools.partial(
                process_single_mseed,
                gmc_df=gmc_df,
                fmax_df=fmax_df,
                bypass_df=bypass_df,
            ),
            mseed_files,
        )

    if not all(value is None for value in skipped_records):
        # Combine the skipped records
        skipped_records = pd.concat(skipped_records)
    else:
        skipped_records = pd.DataFrame(columns=["record_id", "reason"])

    # Save the skipped records
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    skipped_records.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.PROCESSING_SKIPPED_RECORDS,
        index=False,
    )
