"""
fmax calculation
TODO: Needs to be refactored to actually work with the NZGMDB
"""

import functools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from nzgmdb.management import config as cfg


def find_fmaxs(filenames: Iterable[Path], metadata: pd.DataFrame):

    config = cfg.Config()

    # Create empty lists to store fmax values
    fmax_000_list = []
    fmax_090_list = []
    fmax_ver_list = []
    ids = []
    skipped_records = []

    # Iterate over the filenames
    for idx, filename in enumerate(filenames):

        ev_sta_chan = str(filename.stem).replace("_snr_fas", "")[:-1]

        # Get Delta from the metadata
        current_row = metadata.iloc[
            np.where(metadata["ev_sta_chan"] == ev_sta_chan)[0], :
        ]
        ##!! Should explicitely confirm whether this should be (1/a)*b*c or 1/(a*b*c)
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
            (
                snr_all_cols["frequency"]
                >= config.get_value("initial_screening_min_freq_Hz")
            )
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
            ### To do: add to a list of skipped records with the reason such as "failed initial checks"
            # continue

            # TODO replace ev_sta_chan with the updated naming convention
            skipped_records.append(
                {
                    "event_id": ev_sta_chan,
                    "station": ev_sta_chan,
                    "mseed_file": ev_sta_chan,
                    "reason": "File skipped in Fmax initial SNR screening",
                }
            )

        else:
            skipped_records.append(None)

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
        fmax_000_list.append(do_fmax_calc_partial(snr_smooth_gtr_min_freq["snr_000"]))
        fmax_090_list.append(do_fmax_calc_partial(snr_smooth_gtr_min_freq["snr_090"]))
        fmax_ver_list.append(do_fmax_calc_partial(snr_smooth_gtr_min_freq["snr_ver"]))

        ids.append(ev_sta_chan)

    return fmax_000_list, fmax_090_list, fmax_ver_list, ids, skipped_records


def do_fmax_calc(snr_thresh, fny, freq, snr_component):
    """

    Parameters
    ----------
    freq
    snr
    snr_thres
    fny

    Returns
    -------

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
    # Add the ev_sta_chan column to the metadata

    metadata_df["ev_sta_chan"] = (
        metadata_df["evid"] + "_" + metadata_df["sta"] + "_" + metadata_df["chan"]
    )

    # n_procs = 8

    # Split the filenames into n_procs chunks
    snr_filenames = np.array_split(list(snr_filenames), n_procs)

    # with mp.Pool(n_procs) as p:
    #     results = p.map(
    #         functools.partial(
    #             find_fmaxs,
    #             metadata=metadata_df,
    #         ),
    #         snr_filenames,
    #     )
    # Write above as a for loop
    results = []
    for snr_filenames_chunk in snr_filenames:
        results.append(find_fmaxs(snr_filenames_chunk, metadata_df))

    # Combine the results into the 4 different lists
    fmax_000_list = np.concatenate([result[0] for result in results])
    fmax_090_list = np.concatenate([result[1] for result in results])
    fmax_ver_list = np.concatenate([result[2] for result in results])
    ids = np.concatenate([result[3] for result in results])

    skipped_records = [result[4] for result in results]

    # using filter()
    # to remove None values in list
    skipped_records2 = list(filter(lambda item: item is not None, skipped_records[0]))

    skipped_records_df = pd.DataFrame(skipped_records2)

    # Create fmax csv
    fmax = pd.DataFrame(
        {
            "ev_sta": ids,
            "fmax_000": fmax_000_list,
            "fmax_090": fmax_090_list,
            "fmax_ver": fmax_ver_list,
        }
    )

    fmax.to_csv(output_dir / "fmaxA2.csv", index=False)

    skipped_records_df.to_csv(output_dir / "fmax_skipped_records.csv", index=False)
