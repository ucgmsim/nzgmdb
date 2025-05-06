"""
This module contains functions to calculate the SNR for a single mseed file.
"""

import functools
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from IM import im_calculation, snr_calculation
from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors, file_structure
from nzgmdb.mseed_management import reading


def compute_snr_for_single_mseed(
    mseed_file: Path,
    phase_table: pd.DataFrame,
    output_dir: Path,
    ko_directory: Path,
    common_frequency_vector: np.ndarray = im_calculation.DEFAULT_FREQUENCIES,
):
    """
    Compute the SNR for a single mseed file

    Parameters
    ----------
    mseed_file : Path
        Path to the mseed file
    phase_table : pd.DataFrame
        Phase arrival table
    output_dir : Path
        Path to the output directory
    ko_directory : Path
        Path to the directory containing the Ko matrices
    common_frequency_vector : np.ndarray, optional
        Common frequency vector to extract for SNR and FAS, by default None

    Returns
    -------
    meta_df : pd.DataFrame
        Metadata dataframe containing the metadata information for each file
        Can be None if no p-wave arrival could be found or failed to remove sensitivity
        or noise was less than 1 second
    skipped_record : pd.DataFrame
        Dataframe containing the skipped records and reasons why
        Can be None if no p-wave arrival could be found or failed to remove sensitivity
        or noise was less than 1 second

    """
    # Get the station from the filename
    station = mseed_file.name.split("_")[1]

    # Get the event_id
    event_id = file_structure.get_event_id_from_mseed(mseed_file)

    # Read mseed information
    try:
        waveform = reading.create_waveform_from_mseed(mseed_file, pre_process=True, apply_taper=False, apply_zero_padding=False)
    except custom_errors.InventoryNotFoundError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "Failed to find inventory information",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record
    except custom_errors.SensitivityRemovalError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "Failed to remove sensitivity",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record
    except custom_errors.All3ComponentsNotPresentError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "File did not contain 3 components",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record
    except custom_errors.RotationError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "Failed to rotate the data",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record
    except custom_errors.InvalidTraceLengthError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "Invalid trace length from mseed file",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record

    # Get the TP from the phase arrival table
    try:
        tp = phase_table[phase_table["record_id"] == mseed_file.stem][
            "p_wave_ix"
        ].values[0]
    except IndexError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "No P-wave found in phase arrival table",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record

    # Ensure the tp is within the range of the waveform
    stats = reading.read_mseed_to_stream(mseed_file)[0].stats
    if tp > stats.npts or tp < 0:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "TP not in waveform bounds",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            (
                snr,
                fas_signal,
                fas_noise,
                Ds,
                Dn,
            ) = snr_calculation.calculate_snr(
                waveform,
                stats.delta,
                tp,
                frequencies=common_frequency_vector,
                cores=1,
                ko_directory=ko_directory,
            )
    except FileNotFoundError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "Failed to find Ko matrix",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record
    except ValueError:
        skipped_record_dict = {
            "record_id": mseed_file.stem,
            "reason": "Noise was less than 1 second",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record

    # Create dataframe with the snr, fas_noise and fas_signal combined
    # Add prefixes to the column names
    snr = snr.add_prefix("snr_")
    fas_signal = fas_signal.add_prefix("fas_signal_")
    fas_noise = fas_noise.add_prefix("fas_noise_")

    # Concatenate the DataFrames along the columns
    snr_fas_df = pd.concat([snr, fas_signal, fas_noise], axis=1)

    # Save the SNR data
    year_dir = output_dir / str(stats.starttime.year)
    event_dir = year_dir / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    snr_fas_df.to_csv(
        event_dir
        / f"{event_id}_{station}_{stats.channel[:2]}_{stats.location}_snr_fas.csv",
        index_label="frequency",
    )

    # Add to the metadata dataframe
    meta_dict = {
        "record_id": mseed_file.stem,
        "evid": event_id,
        "sta": station,
        "chan": stats.channel[:2],
        "loc": stats.location,
        "tp": tp,
        "Ds": Ds,
        "Dn": Dn,
        "npts": stats.npts,
        "delta": stats.delta,
        "starttime": stats.starttime,
        "endtime": stats.endtime,
    }
    meta_df = pd.DataFrame([meta_dict])
    return meta_df, None


def compute_snr_for_mseed_data(
    data_dir: Path,
    phase_table_path: Path,
    meta_output_dir: Path,
    snr_fas_output_dir: Path,
    ko_directory: Path,
    n_procs: int = 1,
    common_frequency_vector: np.ndarray = None,
    batch_size: int = 5000,
    bypass_records_ffp: Path = None,
):
    """
    Compute the SNR for the data in the data_dir

    Parameters
    ----------
    data_dir : Path
        Directory containing the data for observed waveforms at magnitude bin level structure
    phase_table_path : Path
        Path to the phase arrival table
    meta_output_dir : Path
        Path to the output directory for the metadata and skipped records
    snr_fas_output_dir : Path
        Path to the output directory for the SNR and FAS data
    ko_directory : Path
        Path to the ko matrix directory
    n_procs : int, optional
        Number of processes to use, by default 1
    common_frequency_vector : np.ndarray, optional
        Common frequency vector to use for all the waveforms, by default None
        Uses a default frequency vector if not specified defined in the configuration file
    batch_size : int, optional
        Number of mseed files to process in a single batch, by default 5000
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file, which includes a custom p_wave_ix
    """
    # Create the output directories
    meta_output_dir.mkdir(parents=True, exist_ok=True)
    snr_fas_output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = meta_output_dir / "snr_batch_files"
    batch_dir.mkdir(parents=True, exist_ok=True)

    config = cfg.Config()
    # Creating the common frequency vector if not provided
    if common_frequency_vector is None:
        # Set constants
        common_frequency_start = config.get_value("common_frequency_start")
        common_frequency_end = config.get_value("common_frequency_end")
        common_frequency_num = config.get_value("common_frequency_num")
        common_frequency_vector = np.logspace(
            np.log10(common_frequency_start),
            np.log10(common_frequency_end),
            num=common_frequency_num,
        )

    # Get all the mseed files
    mseed_files = [mseed_file for mseed_file in data_dir.rglob("*.mseed")]

    # Find files that have already been processed and get the suffix indexes
    processed_files = [f for f in batch_dir.iterdir() if f.is_file()]
    processed_suffixes = set(int(f.stem.split("_")[-1]) for f in processed_files)

    print(f"Total number of mseed files to process: {len(mseed_files)}")

    # Create batches of mseed files for checkpointing
    mseed_batches = np.array_split(mseed_files, np.ceil(len(mseed_files) / batch_size))

    # Load the phase arrival table
    phase_table = pd.read_csv(phase_table_path)

    # Load the bypass records if provided
    if bypass_records_ffp is not None:
        bypass_records = pd.read_csv(bypass_records_ffp)
        # remove any records that have a nan p_wave_ix
        bypass_records = bypass_records.dropna(subset=["p_wave_ix"])
        # Identify records in bypass_records that are not in phase_table
        new_records = bypass_records[
            ~bypass_records["record_id"].isin(phase_table["record_id"])
        ]
        # Add these new records to phase_table
        phase_table = pd.concat(
            [phase_table, new_records[["record_id", "p_wave_ix"]]], ignore_index=True
        )
        # Ensure the type of p_wave_ix is int
        phase_table["p_wave_ix"] = phase_table["p_wave_ix"].astype(int)

    for index, batch in enumerate(mseed_batches):
        if index not in processed_suffixes:
            print(f"Processing batch {index + 1}/{len(mseed_batches)}")
            # Process the batch
            with mp.Pool(n_procs) as p:
                results = p.map(
                    functools.partial(
                        compute_snr_for_single_mseed,
                        phase_table=phase_table,
                        output_dir=snr_fas_output_dir,
                        ko_directory=ko_directory,
                        common_frequency_vector=common_frequency_vector,
                    ),
                    batch,
                )

            meta_dfs = []
            skipped_record_dfs = []

            for result in results:
                meta_df, skipped_record = result
                if meta_df is None:
                    skipped_record_dfs.append(skipped_record)
                else:
                    meta_dfs.append(meta_df)

            # Check that there are metadata dataframes that are not None
            if not all(value is None for value in meta_dfs):
                meta_df = pd.concat(meta_dfs, ignore_index=True)
            else:
                print("No metadata dataframes")
                meta_df = pd.DataFrame(
                    columns=[
                        "record_id",
                        "evid",
                        "sta",
                        "chan",
                        "loc",
                        "tp",
                        "Ds",
                        "Dn",
                        "npts",
                        "delta",
                        "starttime",
                        "endtime",
                    ]
                )
            meta_df.to_csv(batch_dir / f"snr_metadata_{index}.csv", index=False)

            # Check that there are skipped records that are not None
            if not all(value is None for value in skipped_record_dfs):
                skipped_records = pd.concat(skipped_record_dfs, ignore_index=True)
                print(f"Skipped {len(skipped_records)} records")
            else:
                print("No skipped records")
                skipped_records = pd.DataFrame(columns=["record_id", "reason"])
            skipped_records.to_csv(
                batch_dir / f"snr_skipped_records_{index}.csv", index=False
            )

    # Join all the batch files
    meta_dfs = []
    skipped_records_dfs = []

    for file in batch_dir.iterdir():
        if "snr_metadata" in file.stem:
            try:
                meta_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")
        elif "snr_skipped_records" in file.stem:
            try:
                skipped_records_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")

    # Check that there are metadata dataframes that are not None
    if not all(value is None for value in meta_dfs):
        meta_df = pd.concat(meta_dfs, ignore_index=True)
    else:
        print("No metadata dataframes")
        meta_df = pd.DataFrame(
            columns=[
                "record_id",
                "evid",
                "sta",
                "chan",
                "loc",
                "tp",
                "Ds",
                "Dn",
                "npts",
                "delta",
                "starttime",
                "endtime",
            ]
        )
    # Check that there are skipped records that are not None
    if not all(value is None for value in skipped_records_dfs):
        skipped_records_df = pd.concat(skipped_records_dfs, ignore_index=True)
        print(f"Skipped {len(skipped_records_df)} records")
    else:
        print("No skipped records")
        skipped_records_df = pd.DataFrame(columns=["record_id", "reason"])

    # Save the dataframes
    meta_df.to_csv(
        meta_output_dir / file_structure.FlatfileNames.SNR_METADATA, index=False
    )
    skipped_records_df.to_csv(
        meta_output_dir / file_structure.SkippedRecordFilenames.SNR_SKIPPED_RECORDS,
        index=False,
    )

    print(f"Finished, output data found in {meta_output_dir}")
