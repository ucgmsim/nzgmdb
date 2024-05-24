"""
fmax calculation ported from a MatLab script

"""

import functools
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from nzgmdb.management import config as cfg


def calculate_fmax(
    snr_thresh: float,
    nyquist_freq_limit: float,
    freq: np.ndarray,
    snr_component: np.ndarray,
) -> float:
    """
    Calculate fmax.

    Parameters
    ----------
    snr_thresh : float
        The maximum SNR for data
        points to be considered.
    nyquist_freq_limit : float
        The Nyquist frequency limit.
    freq : np.ndarray
        1D np.ndarray array of frequency values.
    snr_component : np.ndarray
        1D np.ndarray array of SNR values.
    Returns
    -------
    fmax : float
        Maximum frequency.

    """

    loc = np.where(snr_component < snr_thresh)[0]

    if len(loc) != 0:
        fmax = min(freq[loc[0]], nyquist_freq_limit)
    else:
        fmax = min(freq[-1], nyquist_freq_limit)

    return fmax


def find_fmax(filename: Path, metadata: pd.DataFrame):
    """
    Parameters
    ----------
    filename : Path
        Path to the ___snr_fas.csv file
    metadata : pd.DataFrame
        Contains the SNR meta-data

    Returns
    -------
    fmax_record : dict[str, Any]
        contains record_id, fmax_000, fmax_090, fmax_ver

        skipped_record is a dictionary containing
        record_id, event_id, station, mseed_file, reason
        or None if record was not skipped
    """
    config = cfg.Config()

    record_id = str(filename.stem).replace("_snr_fas", "")

    # Get Delta from the metadata
    current_row = metadata.iloc[np.where(metadata["record_id"] == record_id)[0], :]

    nyquist_freq_limit = (
        (
            1
            / current_row["delta"].iloc[
                config.get_value("nyquist_freq_limit_param_index")
            ]
        )
        * config.get_value("nyquist_freq_limit_param_1")
        * config.get_value("nyquist_freq_limit_param_2")
    )

    # Read CSV file using pandas
    snr_all_cols = pd.read_csv(filename)

    # getting only the snr columns
    snr = snr_all_cols[["snr_000", "snr_090", "snr_ver"]]

    snr_smooth = snr.rolling(
        window=config.get_value("window"),
        center=config.get_value("center"),
        min_periods=config.get_value("min_periods"),
    ).mean()

    # Initial screening:
    # In all components need at least min_points_above_thresh freq points
    # between min_freq_Hz and max_freq_Hz with SNR > snr_thresh_vert
    # or snr_thresh_horiz

    snr_smooth_freq_interval_for_screening = snr_smooth[
        (snr_all_cols["frequency"] >= config.get_value("initial_screening_min_freq_Hz"))
        & (
            snr_all_cols["frequency"]
            <= config.get_value("initial_screening_max_freq_Hz")
        )
    ]

    num_valid_points_in_interval = np.sum(
        snr_smooth_freq_interval_for_screening
        > [
            config.get_value("initial_screening_snr_thresh_horiz"),
            config.get_value("initial_screening_snr_thresh_horiz"),
            config.get_value("initial_screening_snr_thresh_vert"),
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
            "event_id": record_id.split("_")[0],
            "station": record_id.split("_")[1],
            "mseed_file": record_id + ".mseed",
            "reason": "File skipped in fmax initial SNR screening",
        }

    else:
        # keep the record
        skipped_record = None

        snr_smooth_gtr_min_freq = snr_smooth[
            snr_all_cols["frequency"] > config.get_value("min_freq_Hz")
        ]

        freq_gtr_min_freq = (
            snr_all_cols["frequency"]
            .loc[snr_all_cols["frequency"] > config.get_value("min_freq_Hz")]
            .to_numpy()
        )

        calculate_fmax_partial = functools.partial(
            calculate_fmax,
            config.get_value("snr_thresh"),
            nyquist_freq_limit,
            freq_gtr_min_freq,
        )

        fmax_record = {
            "record_id": record_id,
            "fmax_000": calculate_fmax_partial(snr_smooth_gtr_min_freq["snr_000"]),
            "fmax_090": calculate_fmax_partial(snr_smooth_gtr_min_freq["snr_090"]),
            "fmax_ver": calculate_fmax_partial(snr_smooth_gtr_min_freq["snr_ver"]),
        }

    return fmax_record, skipped_record


def start_fmax_calc(
    main_dir: Path, snr_fas_dir: Path, meta_dir: Path, n_procs: int = 1
):
    """

    Parameters
    ----------
    main_dir
    snr_fas_dir
    meta_dir
    n_procs

    Returns
    -------

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

    grouped_results = list(zip(*results))

    fmax_df = pd.DataFrame(grouped_results[0])

    # using filter() to remove None values in list
    skipped_records_df = pd.DataFrame(
        filter(lambda item: item is not None, grouped_results[1])
    )
    print(
        f"Skipped {len(skipped_records_df)} files as their SNR was too low (failed initial screening)"
    )

    fmax_df.to_csv(meta_dir / "fmaxA3.csv", index=False)

    skipped_records_df.to_csv(meta_dir / "fmax_skipped_records.csv", index=False)
