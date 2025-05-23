"""
Calculates the maximum useable frequency (fmax).
"""

import functools
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from nzgmdb.mseed_management import reading


def run_full_fmax_calc(
    meta_output_dir: Path,
    waveform_dir: Path,
    snr_fas_output_dir: Path,
    n_procs: int = 1,
    bypass_records_ffp: Path = None,
):
    """
    Run the full procedure for each record to assess SNR produced from mseed files
    and get the maximum usable frequency (fmax).

    Parameters
    ----------
    meta_output_dir : Path
        Path to the output directory for the metadata and skipped records.
    waveform_dir : Path
        Path to the directory containing the mseed files to process.
    snr_fas_output_dir : Path
        Path to the output directory for the SNR and FAS data.
    n_procs : int, optional
        Number of processes to use, by default 1.
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file, which includes a custom fmax
    """
    mseed_files = list(waveform_dir.rglob("*.mseed"))

    with mp.Pool(n_procs) as p:
        results = p.map(
            functools.partial(
                assess_snr_and_get_fmax,
                snr_fas_output_dir=snr_fas_output_dir,
            ),
            mseed_files,
        )

    if len(results) == 0:
        print("No records to process")
        meta_dfs, skipped_record_dfs = [None], [None]
    else:
        # Unpack the results
        meta_dfs, skipped_record_dfs = zip(*results)

    # Check that there are dataframes that are not None before concatenating
    if not all(value is None for value in meta_dfs):
        fmax_df = pd.concat(meta_dfs).reset_index(drop=True)
    else:
        fmax_df = pd.DataFrame(
            columns=["record_id", "fmax_000", "fmax_090", "fmax_ver"]
        )
    if not all(value is None for value in skipped_record_dfs):
        skipped_records_df = pd.concat(skipped_record_dfs).reset_index(drop=True)
    else:
        skipped_records_df = pd.DataFrame(columns=["record_id", "reason"])

    print(f"Skipped {len(skipped_records_df)} records")
    skipped_records_df.to_csv(
        meta_output_dir / file_structure.SkippedRecordFilenames.FMAX_SKIPPED_RECORDS,
        index=False,
    )

    # Check if the bypass records file is provided
    if bypass_records_ffp is not None:
        # Add the bypass records to the fmax_df, or overwrite the fmax values if the record is in the bypass records
        bypass_df = pd.read_csv(bypass_records_ffp)
        fmax_df = fmax_df.merge(
            bypass_df[["record_id", "fmax_000", "fmax_090", "fmax_ver"]],
            on="record_id",
            how="left",
            suffixes=("", "_bypass"),
        )
        fmax_df["fmax_000"] = fmax_df["fmax_000_bypass"].fillna(fmax_df["fmax_000"])
        fmax_df["fmax_090"] = fmax_df["fmax_090_bypass"].fillna(fmax_df["fmax_090"])
        fmax_df["fmax_ver"] = fmax_df["fmax_ver_bypass"].fillna(fmax_df["fmax_ver"])
        fmax_df = fmax_df.drop(
            columns=[
                "fmax_000_bypass",
                "fmax_090_bypass",
                "fmax_ver_bypass",
            ]
        )

    fmax_df.to_csv(meta_output_dir / file_structure.FlatfileNames.FMAX, index=False)


def assess_snr_and_get_fmax(
    filename: Path, snr_fas_output_dir: Path
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Assess the record's Signal-to-Noise Ratio (SNR) within the frequency interval specified in the config file
    and determine the record's maximum usable frequency (fmax) if its SNR exceeds the threshold.

    Parameters
    ----------
    filename : Path
        Path to the `record_id.mseed` file.
    snr_fas_output_dir : Path
        Path to the output directory where the `record_id_snr_fas.csv` file is stored.

    Returns
    -------
    fmax_record: pd.DataFrame
        DataFrame containing the fmax values, if the record is skipped the values are set to the Nyquist frequency.
    skipped_record: pd.DataFrame | None
        If the record is skipped, returns a DataFrame containing the reason for skipping, otherwise returns None.
    """
    config = cfg.Config()
    record_id = filename.stem

    # read the mseed file to get the delta
    mseed = reading.read_mseed_to_stream(filename)
    dt = mseed[0].stats.delta

    # current_row["delta"] is a pd.Series() containing 1 float so .iloc[0]
    # is used to get the float from the pd.Series()
    scaled_nyquist_freq = (
        (1 / dt) * 0.5 * config.get_value("nyquist_freq_scaling_factor")
    )

    snr_filename = next(snr_fas_output_dir.rglob(f"{record_id}_snr_fas.csv"), None)
    if snr_filename is None:
        # Use the Nyquist frequency as the fmax
        fmax_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "fmax_000": scaled_nyquist_freq,
                    "fmax_090": scaled_nyquist_freq,
                    "fmax_ver": scaled_nyquist_freq,
                }
            ]
        )
        skipped_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "reason": "No SNR file found for record",
                }
            ]
        )
        return fmax_record, skipped_record

    # Assess the SNR
    snr_with_freq_signal_noise = pd.read_csv(snr_filename)
    snr = snr_with_freq_signal_noise[["snr_000", "snr_090", "snr_ver"]]

    snr_smooth = snr.rolling(
        window=config.get_value("window"),
        center=config.get_value("center"),
        min_periods=config.get_value("min_periods"),
    ).mean()

    # SNR assessment:
    # Need at least min_points_above_thresh points
    # in the interval initial_screening_min_freq_Hz to
    # initial_screening_max_freq_Hz
    # with SNR > initial_screening_snr_thresh_ver
    # for the ver component and SNR > initial_screening_snr_thresh_horiz
    # for the 000 and 090 components

    snr_smooth_freq_interval_for_screening = snr_smooth.loc[
        (
            snr_with_freq_signal_noise["frequency"]
            >= config.get_value("initial_screening_min_freq_Hz")
        )
        & (
            snr_with_freq_signal_noise["frequency"]
            <= config.get_value("initial_screening_max_freq_Hz")
        )
    ]

    num_valid_points_in_interval = np.sum(
        snr_smooth_freq_interval_for_screening
        > [
            config.get_value("initial_screening_snr_thresh_horiz"),
            config.get_value("initial_screening_snr_thresh_horiz"),
            config.get_value("initial_screening_snr_thresh_ver"),
        ],
        axis=0,
    )

    if (
        num_valid_points_in_interval
        > config.get_value("initial_screening_min_points_above_thresh")
    ).all():
        # keep the record
        skipped_record = None

        snr_smooth_gtr_min_freq = snr_smooth.loc[
            snr_with_freq_signal_noise["frequency"] > config.get_value("min_freq_Hz")
        ]

        freq_gtr_min_freq = (
            snr_with_freq_signal_noise["frequency"]
            .loc[
                snr_with_freq_signal_noise["frequency"]
                > config.get_value("min_freq_Hz")
            ]
            .to_numpy()
        )

        calculate_fmax_partial = functools.partial(
            calculate_fmax,
            config.get_value("snr_thresh"),
            scaled_nyquist_freq,
            freq_gtr_min_freq,
        )

        fmax_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "fmax_000": calculate_fmax_partial(
                        snr_smooth_gtr_min_freq["snr_000"]
                    ),
                    "fmax_090": calculate_fmax_partial(
                        snr_smooth_gtr_min_freq["snr_090"]
                    ),
                    "fmax_ver": calculate_fmax_partial(
                        snr_smooth_gtr_min_freq["snr_ver"]
                    ),
                }
            ]
        )

        ############################################################################

    else:
        # Use the Nyquist frequency as the fmax
        fmax_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "fmax_000": scaled_nyquist_freq,
                    "fmax_090": scaled_nyquist_freq,
                    "fmax_ver": scaled_nyquist_freq,
                }
            ]
        )

        skipped_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "reason": (
                        f"Record skipped in fmax calculation because there were not at least "
                        f"{config.get_value('initial_screening_min_points_above_thresh')} frequency points in the interval "
                        f"{config.get_value('initial_screening_min_freq_Hz')} - "
                        f"{config.get_value('initial_screening_max_freq_Hz')} Hz "
                        f"with SNR > {config.get_value('initial_screening_snr_thresh_ver')} for the ver component and "
                        f"SNR > {config.get_value('initial_screening_snr_thresh_horiz')} for the 000 and 090 components."
                    ),
                }
            ]
        )

    return fmax_record, skipped_record


def calculate_fmax(
    snr_thresh: float,
    scaled_nyquist_freq: float,
    freq: np.ndarray,
    snr: np.ndarray,
) -> float:
    """
    Calculates the maximum usable frequency (fmax).

    Parameters
    ----------
    snr_thresh : float
        The SNR threshold for data
        points to be considered.
    scaled_nyquist_freq : float
        The Nyquist frequency scaled by
        a factor of nyquist_freq_scaling_factor
        in the config file.
    freq : np.ndarray
        1D array of frequency values.
    snr : np.ndarray
        1D of SNR values.

    Returns
    -------
    float
        Maximum frequency.

    """

    loc = np.where(snr < snr_thresh)[0]

    if len(loc) > 0:
        fmax = min(freq[loc[0]], scaled_nyquist_freq)
    else:
        fmax = min(freq[-1], scaled_nyquist_freq)

    return fmax
