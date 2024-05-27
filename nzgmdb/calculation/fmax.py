"""
Calculates the maximum useable frequency (fmax).
"""

import functools
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nzgmdb.management import config as cfg


def start_fmax_calc(
    main_dir: Path, meta_dir: Path, snr_fas_dir: Path, n_procs: int = 1
):
    """
    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (highest level directory)
        (glob is used to find all mseed files recursively)
    meta_output_dir : Path
        Path to the directory for the metadata and skipped records
    snr_fas_output_dir : Path
        Path to the directory for the SNR and FAS data
    n_procs : int, optional
        Number of processes to use, by default 1
    """

    if not meta_dir:
        meta_dir = main_dir / "flatfiles"

    if not snr_fas_dir:
        snr_fas_dir = Path(main_dir / "snr_fas")

    metadata_df = pd.read_csv(meta_dir / "snr_metadata.csv")
    snr_filenames = snr_fas_dir.glob("**/*snr_fas.csv")

    with multiprocessing.Pool(n_procs) as pool:
        results = pool.map(
            functools.partial(
                find_fmax,
                metadata=metadata_df,
            ),
            snr_filenames,
        )

    # using filter() to remove None values
    skipped_records_df = pd.DataFrame(
        filter(lambda item: item is not None, list(zip(*results))[1])
    )
    print(f"Skipped {len(skipped_records_df)} records")

    fmax_df = pd.DataFrame(
        filter(lambda item: item is not None, list(zip(*results))[0])
    )

    fmax_df.to_csv(meta_dir / "fmax.csv", index=False)

    skipped_records_df.to_csv(meta_dir / "fmax_skipped_records.csv", index=False)


def find_fmax(
    filename: Path, metadata: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Parameters
    ----------
    filename : Path
        Path to the ___snr_fas.csv file
    metadata : pd.DataFrame
        Contains the SNR meta data

    Returns
    -------
    fmax_record : dict[str, Any] or None if record
                  is not skipped or skipped, respectively
    if not None, contains record_id, fmax_000, fmax_090, fmax_ver

    skipped_record : dict[str, Any] or None if record
                    is skipped or not skipped, respectively
    if not None, contains record_id, reason
    """
    config = cfg.Config()

    record_id = str(filename.stem).replace("_snr_fas", "")

    # Get delta from the metadata
    current_row = metadata.iloc[np.where(metadata["record_id"] == record_id)[0], :]

    # current_row["delta"] is a pd.Series() containing 1 float so .iloc[0]
    # is used to get the float from the pd.Series()
    scaled_nyquist_freq = (
        (1 / current_row["delta"].iloc[0])
        * 0.5
        * config.get_value("nyquist_freq_scaling_factor")
    )

    snr_with_other_cols = pd.read_csv(filename)
    snr = snr_with_other_cols[["snr_000", "snr_090", "snr_ver"]]

    snr_smooth = snr.rolling(
        window=config.get_value("window"),
        center=config.get_value("center"),
        min_periods=config.get_value("min_periods"),
    ).mean()

    # Initial screening:
    # Need at least min_points_above_thresh points
    # in the interval initial_screening_min_freq_Hz to
    # initial_screening_max_freq_Hz
    # with SNR > initial_screening_snr_thresh_ver
    # for the ver component and SNR > initial_screening_snr_thresh_horiz
    # for the 000 and 090 components

    snr_smooth_freq_interval_for_screening = snr_smooth.loc[
        (
            snr_with_other_cols["frequency"]
            >= config.get_value("initial_screening_min_freq_Hz")
        )
        & (
            snr_with_other_cols["frequency"]
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

    if not (
        num_valid_points_in_interval
        > config.get_value("initial_screening_min_points_above_thresh")
    ).all():
        # Skip the record
        fmax_record = None

        skipped_record = {
            "record_id": record_id,
            "reason": (
                f"Record skipped in fmax initial screening because "
                f"there were not at least "
                f"{config.get_value('initial_screening_min_points_above_thresh')} "
                f"frequency points in the interval "
                f"{config.get_value('initial_screening_min_freq_Hz')} - "
                f"{config.get_value('initial_screening_max_freq_Hz')} Hz "
                f"with SNR > {config.get_value('initial_screening_snr_thresh_ver')} "
                f"for the ver component and SNR > {config.get_value('initial_screening_snr_thresh_horiz')} "
                f"for the 000 and 090 components."
            ),
        }

    else:
        # keep the record
        skipped_record = None

        snr_smooth_gtr_min_freq = snr_smooth.loc[
            snr_with_other_cols["frequency"] > config.get_value("min_freq_Hz")
        ]

        freq_gtr_min_freq = (
            snr_with_other_cols["frequency"]
            .loc[snr_with_other_cols["frequency"] > config.get_value("min_freq_Hz")]
            .to_numpy()
        )

        calculate_fmax_partial = functools.partial(
            calculate_fmax,
            config.get_value("snr_thresh"),
            scaled_nyquist_freq,
            freq_gtr_min_freq,
        )

        fmax_record = {
            "record_id": record_id,
            "fmax_000": calculate_fmax_partial(snr_smooth_gtr_min_freq["snr_000"]),
            "fmax_090": calculate_fmax_partial(snr_smooth_gtr_min_freq["snr_090"]),
            "fmax_ver": calculate_fmax_partial(snr_smooth_gtr_min_freq["snr_ver"]),
        }

    return fmax_record, skipped_record


def calculate_fmax(
    snr_thresh: float,
    scaled_nyquist_freq: float,
    freq: np.ndarray,
    snr: np.ndarray,
) -> float:
    """
    Calculate fmax.

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
    fmax : float
        Maximum frequency.

    """

    loc = np.where(snr < snr_thresh)[0]

    if len(loc) != 0:
        fmax = min(freq[loc[0]], scaled_nyquist_freq)
    else:
        fmax = min(freq[-1], scaled_nyquist_freq)

    return fmax
