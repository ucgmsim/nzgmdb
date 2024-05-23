"""
fmax calculation
TODO: Needs to be refactored to actually work with the NZGMDB
"""

import functools
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from nzgmdb.management import config as cfg


def find_fmax(filename: Path, metadata: pd.DataFrame):

    config = cfg.Config()

    record_id = str(filename.stem).replace("_snr_fas", "")

    # Get Delta from the metadata
    current_row = metadata.iloc[np.where(metadata["record_id"] == record_id)[0], :]

    ##!! TODO Should explicitely confirm whether this should be (1/a)*b*c or 1/(a*b*c)
    fny = (
        1
        / current_row["delta"].iloc[config.get_value("fny_param_index")]
        * config.get_value("fny_param_1")
        * config.get_value("fny_param_2")
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
    # In all components at least min_points_above_thresh freq points
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

        skipped_record = {
            "record_id": record_id,
            "event_id": record_id.split("_")[0],
            "station": record_id.split("_")[1],
            "mseed_file": record_id + ".mseed",
            "reason": "File skipped in fmax initial SNR screening",
        }

    else:
        skipped_record = None

    ##############################################

    snr_smooth_gtr_min_freq = snr_smooth[
        snr_all_cols["frequency"] > config.get_value("min_freq_Hz")
    ]

    freq_gtr_min_freq = (
        snr_all_cols["frequency"]
        .loc[snr_all_cols["frequency"] > config.get_value("min_freq_Hz")]
        .to_numpy()
    )

    do_fmax_calc_partial = functools.partial(
        do_fmax_calc, config.get_value("snr_thresh"), fny, freq_gtr_min_freq
    )
    # Compute fmax for each component
    fmax_000 = do_fmax_calc_partial(snr_smooth_gtr_min_freq["snr_000"])
    fmax_090 = do_fmax_calc_partial(snr_smooth_gtr_min_freq["snr_090"])
    fmax_ver = do_fmax_calc_partial(snr_smooth_gtr_min_freq["snr_ver"])

    return record_id, fmax_000, fmax_090, fmax_ver, skipped_record


def do_fmax_calc(
    snr_thresh: float, fny: float, freq: np.ndarray, snr_component: np.ndarray
) -> float:
    """
    Calculate fmax.

    Parameters
    ----------
    snr_thresh : float
        The maximum SNR for considered
        data point
    fny : float
        TODO What is this parameter?
        ???????????
    freq : np.ndarray
        1D np.ndarray array of frequency values
    snr_component : np.ndarray
        1D np.ndarray array of SNR values
    Returns
    -------
    fmax : float
        Maximum frequency

    """

    loc = np.where(snr_component < snr_thresh)[0]

    if len(loc) != 0:
        fmax = min(freq[loc[0]], fny)
    else:
        fmax = min(freq[-1], fny)

    return fmax


def temp_fmax_call_func(main_dir: Path, n_procs):

    output_dir = main_dir / "flatfiles"

    # # File inputs
    # metadata_df = pd.read_csv(
    #     "/home/arr65/gmdb_A/new_struct_2022_2/flatfiles/snr_metadata.csv"
    # )
    snr_dir = Path(main_dir / "snr_fas")

    metadata_df = pd.read_csv(output_dir / "snr_metadata.csv")

    snr_filenames = snr_dir.glob("**/*snr_fas.csv")

    metadata_df["record_id"] = (
        metadata_df["evid"]
        + "_"
        + metadata_df["sta"]
        + "_"
        + metadata_df["chan"]
        + "_"
        + metadata_df["loc"]
    )

    with multiprocessing.Pool(n_procs) as pool:
        results = pool.map(
            functools.partial(
                find_fmax,
                metadata=metadata_df,
            ),
            snr_filenames,
        )

    grouped_results = list(zip(*results))

    fmax = pd.DataFrame(
        {
            "record_id": grouped_results[0],
            "fmax_000": grouped_results[1],
            "fmax_090": grouped_results[2],
            "fmax_ver": grouped_results[3],
        }
    )

    # using filter() to remove None values in list
    skipped_records_df = pd.DataFrame(
        filter(lambda item: item is not None, grouped_results[4])
    )

    fmax.to_csv(output_dir / "fmaxA2.csv", index=False)

    skipped_records_df.to_csv(output_dir / "fmax_skipped_records.csv", index=False)
