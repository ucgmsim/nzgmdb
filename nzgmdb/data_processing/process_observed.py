"""
This module contains functions to process observed data from mseed files and turn them into ascii files
"""

import functools
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read_inventory

import qcore.timeseries as ts
from nzgmdb.data_processing import waveform_manipulation
from nzgmdb.management import custom_errors, file_structure
from nzgmdb.mseed_management import reading


def process_single_mseed(
    mseed_file: Path,
    gmc_df: pd.DataFrame | None = None,
    fmax_df: pd.DataFrame | None = None,
    bypass_df: pd.DataFrame | None = None,
    xml_dir: Path | None = None,
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
    gmc_df : pd.DataFrame, optional
        The GMC values containing fmin information
    fmax_df : pd.DataFrame, optional
        The Fmax values
    bypass_df : pd.DataFrame, optional
        The bypass records containing custom fmin, fmax values
    xml_dir : Path, optional
        The directory containing the station xml files for inventory information

    Returns
    -------
    pd.DataFrame | None
        Dataframe containing the skipped record name and reason why
        or None if the record was processed successfully
    """
    # Check if the mseed file is in the GMC predictions
    mseed_stem = mseed_file.stem
    gmc_rows = None if gmc_df is None else gmc_df[gmc_df["record"] == mseed_stem]

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

    inventory = None
    if xml_dir:
        # Load the inventory information
        inventory_file = xml_dir / f"{station}.xml"
        if inventory_file.is_file():
            inventory = read_inventory(inventory_file)

    # Perform initial pre-processing
    try:
        mseed = waveform_manipulation.initial_preprocessing(mseed, inventory=inventory)
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
    except custom_errors.DiffrentiateError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "Unable to differentiate record",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record
    except custom_errors.DetrendError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "Unable to detrend record",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Get the GMC fmin values
    fmin_h = (
        None
        if gmc_rows is None or gmc_rows.empty
        else gmc_rows[gmc_rows["component"].isin(["X", "Y"])]["fmin_mean"].max()
    )
    fmin_v = (
        None
        if gmc_rows is None or gmc_rows.empty
        else gmc_rows[gmc_rows["component"] == "Z"]["fmin_mean"].iloc[0]
    )

    # Get the fmax values
    fmax_rows = None if fmax_df is None else fmax_df[fmax_df["record_id"] == mseed_stem]
    fmax_h = (
        None
        if fmax_df is None or fmax_rows.empty
        else min(fmax_rows.loc[:, ["fmax_000", "fmax_090"]].values[0])
    )
    fmax_v = (
        None if fmax_df is None or fmax_rows.empty else fmax_rows["fmax_ver"].iloc[0]
    )

    # Check if the record is in the bypass records
    if bypass_df is not None:
        if mseed_stem in bypass_df["record_id"].values:
            bypass_row_data = bypass_df.loc[bypass_df["record_id"] == mseed_stem].iloc[
                0
            ]
            fmin_bypass_h = bypass_row_data[["fmin_000", "fmin_090"]].max()
            fmin_bypass_v = bypass_row_data["fmin_ver"]
            fmax_bypass_h = bypass_row_data[["fmax_000", "fmax_090"]].min()
            fmax_bypass_v = bypass_row_data["fmax_ver"]
            fmin_h = fmin_h if np.isnan(fmin_bypass_h) else fmin_bypass_h
            fmin_v = fmin_v if np.isnan(fmin_bypass_v) else fmin_bypass_v
            fmax_h = fmax_h if np.isnan(fmax_bypass_h) else fmax_bypass_h
            fmax_v = fmax_v if np.isnan(fmax_bypass_v) else fmax_bypass_v

    # Perform high and lowcut processing
    try:
        (
            acc_bb_000,
            acc_bb_090,
            acc_bb_ver,
        ) = waveform_manipulation.high_and_low_cut_processing(
            mseed, dt, fmin_h, fmin_v, fmax_h, fmax_v
        )
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
    gmc_ffp: Path | None = None,
    fmax_ffp: Path | None = None,
    bypass_records_ffp: Path | None = None,
    n_procs: int = 1,
):
    """
    Process the mseed files to txt files
    Saves the skipped records to a csv file and gives reasons why they were skipped

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    gmc_ffp : Path, optional
        The full file path to the GMC predictions file
    fmax_ffp : Path, optional
        The full file path to the Fmax file
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file, which includes a custom fmin, fmax
    n_procs : int, optional
        The number of processes to use for multiprocessing
    """
    # Get the raw waveform mseed files
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    xml_dir = file_structure.get_stationxml_dir(main_dir)
    mseed_files = waveform_dir.rglob("*.mseed")

    # Load the GMC, Fmax and bypass records
    gmc_df = None if gmc_ffp is None else pd.read_csv(gmc_ffp)
    try:
        fmax_df = None if fmax_ffp is None else pd.read_csv(fmax_ffp)
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
                xml_dir=xml_dir,
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
