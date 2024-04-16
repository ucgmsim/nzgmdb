import argparse
import functools
import multiprocessing as mp
from pathlib import Path

import pytz
import obspy
import numpy as np
import pandas as pd
from obspy.core.trace import Stats
from pandas._libs.tslibs.timestamps import Timestamp

import IM_calculation.IM.snr_calculation as snr_calc
from IM_calculation.IM.read_waveform import create_waveform_from_data

from nzgmdb.management import file_structure, config as cfg
from nzgmdb.mseed_management import reading


def create_waveform_from_files(
    comp_000_file: Path, comp_090_file: Path, comp_ver_file: Path, nt: int, dt: float
):
    """
    Read the waveform from the files and creates a waveform object
    Note: Will return None if there are inf or nan values in the middle of the ASCII files

    Parameters
    ----------
    comp_000_file : Path
        Path to the file containing the 000 component
    comp_090_file : Path
        Path to the file containing the 090 component
    comp_ver_file : Path
        Path to the file containing the ver component
    nt : int
        Number of samples
    dt : float
        Sampling rate
    """
    comp_000 = pd.read_csv(
        comp_000_file, sep="\s+", header=None, skiprows=2
    ).values.ravel()
    comp_090 = pd.read_csv(
        comp_090_file, sep="\s+", header=None, skiprows=2
    ).values.ravel()
    comp_ver = pd.read_csv(
        comp_ver_file, sep="\s+", header=None, skiprows=2
    ).values.ravel()

    # Find the index of the last non-inf and non-nan value for each comp
    comp_000_idx = len(comp_000) - np.argmax(
        (comp_000[::-1] != np.inf) & (~np.isnan(comp_000[::-1]))
    )
    comp_090_idx = len(comp_090) - np.argmax(
        (comp_090[::-1] != np.inf) & (~np.isnan(comp_090[::-1]))
    )
    comp_ver_idx = len(comp_ver) - np.argmax(
        (comp_ver[::-1] != np.inf) & (~np.isnan(comp_ver[::-1]))
    )

    # Find the smalled index
    min_idx = min(comp_000_idx, comp_090_idx, comp_ver_idx)

    # Remove inf and nan values from the end
    comp_000 = comp_000[:min_idx]
    comp_090 = comp_090[:min_idx]
    comp_ver = comp_ver[:min_idx]

    # Check for any inf values or nans
    if np.any(np.isinf(comp_000)) or np.any(np.isnan(comp_000)):
        return None
    if np.any(np.isinf(comp_090)) or np.any(np.isnan(comp_090)):
        return None
    if np.any(np.isinf(comp_ver)) or np.any(np.isnan(comp_ver)):
        return None

    # Create the waveform object
    waveform = create_waveform_from_data(
        np.stack((comp_090, comp_000, comp_ver), axis=1), NT=nt, DT=dt
    )

    return waveform


def get_tp_from_phase_table(phase_table_path: Path, mseed_stats: Stats, event_id: str):
    """
    Get the time of the p-wave arrival from the phase arrival table
    Using the event id and the mseed file to gather metadata

    Parameters
    ----------
    phase_table_path : Path
        Path to the phase arrival table
    mseed_stats : Stats
        Stats object from the mseed file
    event_id : str
        Event id of the event

    Returns
    -------
    tp: Index of the p-wave arrival for the waveform array,
        None if no p-wave arrival could be found
    """
    phase_arrival_table = pd.read_csv(phase_table_path, low_memory=False)

    # Getting the appropriate row from the phase arrival table
    event_df = phase_arrival_table.loc[
        (phase_arrival_table["evid"] == event_id)
        & (phase_arrival_table["sta"] == mseed_stats.station)
    ]
    # If there is no data for the event / station pair, return None
    if len(event_df) == 0:
        return None

    # Select the correct phase (Needs to start with a P)
    phase_row = event_df.loc[event_df["phase"].str.upper().str.startswith("P")]

    # If there is no P phase, return None
    if len(phase_row) == 0:
        return None

    # Get the time of the P phase
    tp_time = Timestamp(phase_row["datetime"].values[0], tz=pytz.UTC)

    # Datetime conversions to timestamps for comparisions with and without UTC
    dt_start = mseed_stats.starttime
    dt_end = mseed_stats.endtime
    dt_start_UTC = Timestamp(
        year=dt_start.year,
        month=dt_start.month,
        day=dt_start.day,
        hour=dt_start.hour,
        minute=dt_start.minute,
        second=dt_start.second,
        microsecond=dt_start.microsecond,
        tz=pytz.UTC,
    )
    dt_end_UTC = Timestamp(
        year=dt_end.year,
        month=dt_end.month,
        day=dt_end.day,
        hour=dt_end.hour,
        minute=dt_end.minute,
        second=dt_end.second,
        microsecond=dt_end.microsecond,
        tz=pytz.UTC,
    )

    # Calculate the time difference between the start and end times
    time_diff = dt_end_UTC - dt_start_UTC

    # Calculate the time difference between each point
    interval = time_diff / (mseed_stats.npts - 1)

    # Calculate the position of the tp time within the time range
    desired_position = (tp_time - dt_start_UTC) / interval

    # Round the desired position to the nearest whole number
    tp = round(desired_position)

    # Ensure the tp is within the range of the waveform
    if tp > mseed_stats.npts or tp < 0:
        return None
    else:
        return tp


def compute_snr_for_single_mseed(
    mseed_file: Path,
    phase_table_path: Path,
    output_dir: Path,
    apply_smoothing: bool = True,
    ko_matrix_path: Path = None,
    common_frequency_vector: np.ndarray = None,
):
    """
    Compute the SNR for a single mseed file

    Parameters
    ----------
    mseed_file : Path
        Path to the mseed file
    phase_table_path : Path
        Path to the phase arrival table
    output_dir : Path
        Path to the output directory
    apply_smoothing : bool, optional
        Whether to apply smoothing to the SNR calculation, by default True
    ko_matrix_path : Path, optional
        Path to the ko matrix, by default None
    common_frequency_vector : np.ndarray, optional
        Common frequency vector to extract for SNR and FAS, by default None

    Returns
    -------
    (meta_df, skipped_record_df): Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing the metadata dataframe and the skipped record dataframe,
        Can be None if no p-wave arrival could be found or if the ASCII files are missing
    """
    skipped_record = None

    # Get the station from the filename
    station = mseed_file.name.split("_")[1]

    # Get the event_id
    event_id = file_structure.get_event_id_from_mseed(mseed_file)

    # Read mseed information
    waveform = reading.create_waveform_from_mseed(mseed_file)
    stats = obspy.read(str(mseed_file))[0].stats

    if waveform is None:
        skipped_record_dict = {
            "event_id": event_id,
            "station": station,
            "mseed_file": mseed_file.name,
            "reason": "Inf or NaN values in middle of ASCII component files",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record

    # Index of the start of the P-wave
    tp = get_tp_from_phase_table(phase_table_path, stats, event_id)

    # If there is no tp, record and skip this file
    if tp is None:
        skipped_record_dict = {
            "event_id": event_id,
            "station": station,
            "mseed_file": mseed_file.name,
            "reason": "No P-wave found in phase arrival table",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record

    (
        snr,
        frequencies,
        fas_signal,
        fas_noise,
        Ds,
        Dn,
    ) = snr_calc.get_snr_from_waveform(
        waveform,
        tp,
        apply_smoothing=apply_smoothing,
        ko_matrix_path=ko_matrix_path,
        common_frequency_vector=common_frequency_vector,
        sampling_rate=stats.sampling_rate,
    )

    if snr is None:
        skipped_record_dict = {
            "event_id": event_id,
            "station": station,
            "mseed_file": mseed_file.name,
            "reason": "Noise was less than 1 second",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record

    # Save the SNR data
    # Create dataframe with the snr, fas_noise and fas_signal
    snr_fas_df = pd.DataFrame(
        {
            "snr_000": snr[:, 0],
            "snr_090": snr[:, 1],
            "snr_ver": snr[:, 2],
            "fas_signal_000": fas_signal[:, 0],
            "fas_signal_090": fas_signal[:, 1],
            "fas_signal_ver": fas_signal[:, 2],
            "fas_noise_000": fas_noise[:, 0],
            "fas_noise_090": fas_noise[:, 1],
            "fas_noise_ver": fas_noise[:, 2],
        },
        index=frequencies,
    )
    year_dir = output_dir / str(stats.starttime.year)
    event_dir = year_dir / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    snr_fas_df.to_csv(
        event_dir / f"{event_id}_{station}_{stats.channel}_snr_fas.csv",
        index_label="frequency",
    )
    # Add to the metadata dataframe
    meta_dict = {
        "evid": event_id,
        "sta": station,
        "chan": stats.channel,
        "tp": tp,
        "Ds": Ds,
        "Dn": Dn,
        "npts": stats.npts,
        "delta": stats.delta,
        "starttime": stats.starttime,
        "endtime": stats.endtime,
    }
    meta_df = pd.DataFrame([meta_dict])
    return meta_df, skipped_record


def compute_snr_for_mseed_data(
    data_dir: Path,
    phase_table_path: Path,
    meta_output_dir: Path,
    snr_fas_output_dir: Path,
    n_procs: int = 1,
    apply_smoothing: bool = True,
    ko_matrix_path: Path = None,
    common_frequency_vector: np.ndarray = None,
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
    n_procs : int, optional
        Number of processes to use, by default 1
    apply_smoothing : bool, optional
        Whether to apply smoothing to the SNR calculation, by default True
    ko_matrix_path : Path, optional
        Path to the ko matrix, by default None
    common_frequency_vector : np.ndarray, optional
        Common frequency vector to use for all the waveforms, by default None
        Uses a default frequency vector if not specified defined in the configuration file
    """
    # Create the output directories
    meta_output_dir.mkdir(parents=True, exist_ok=True)
    snr_fas_output_dir.mkdir(parents=True, exist_ok=True)

    # Creating the common frequency vector if not provided
    if common_frequency_vector is None:
        config = cfg.Config()
        common_frequency_vector = np.logspace(np.log10(0.01318257), np.log10(100), num=389)

    # Get all the mseed files
    mseed_files = [mseed_file for mseed_file in data_dir.glob("**/*.mseed")]

    print(f"Total number of mseed files to process: {len(mseed_files)}")

    # Plot using multiprocessing all that are added to the plot_queue
    with mp.Pool(n_procs) as p:
        df_rows = p.map(
            functools.partial(
                compute_snr_for_single_mseed,
                phase_table_path=phase_table_path,
                output_dir=snr_fas_output_dir,
                apply_smoothing=apply_smoothing,
                ko_matrix_path=ko_matrix_path,
                common_frequency_vector=common_frequency_vector,
            ),
            mseed_files,
        )

    # Unpack the results
    meta_dfs, skipped_record_dfs = zip(*df_rows)

    # Check that there are metadata dataframes that are not None
    if not all(value is None for value in meta_dfs):
        meta_df = pd.concat(meta_dfs).reset_index(drop=True)
    else:
        print("No metadata dataframes")
        meta_df = pd.DataFrame()
    meta_df.to_csv(meta_output_dir / "snr_metadata.csv", index=False)

    # Check that there are skipped records that are not None
    if not all(value is None for value in skipped_record_dfs):
        skipped_records = pd.concat(skipped_record_dfs).reset_index(drop=True)
        print(f"Skipped {len(skipped_records)} records")
    else:
        print("No skipped records")
        skipped_records = pd.DataFrame()
    skipped_records.to_csv(meta_output_dir / "snr_skipped_records.csv", index=False)

    print(f"Finished, output data found in {meta_output_dir}")
