import functools
import multiprocessing as mp
from pathlib import Path

import obspy
import numpy as np
import pandas as pd

import IM_calculation.IM.snr_calculation as snr_calc
from nzgmdb.management import file_structure, config as cfg
from nzgmdb.mseed_management import reading
from nzgmdb.phase_arrival import tp_selection


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
    waveform = reading.create_waveform_from_mseed(mseed_file, process=True)

    if waveform is None:
        skipped_record_dict = {
            "event_id": event_id,
            "station": station,
            "mseed_file": mseed_file.name,
            "reason": "Failed to remove sensitivity",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return None, skipped_record

    stats = obspy.read(str(mseed_file))[0].stats

    # Index of the start of the P-wave
    tp = tp_selection.get_tp_from_phase_table(phase_table_path, stats, event_id)

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
        "chan": stats.channel[:2],
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
        # Set constants
        config = cfg.Config()
        common_frequency_start = config.get_value("common_frequency_start")
        common_frequency_end = config.get_value("common_frequency_end")
        common_frequency_num = config.get_value("common_frequency_num")
        common_frequency_vector = np.logspace(
            np.log10(common_frequency_start),
            np.log10(common_frequency_end),
            num=common_frequency_num,
        )

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
