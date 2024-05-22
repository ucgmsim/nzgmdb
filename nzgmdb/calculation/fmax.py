"""
fmax calculation
TODO: Needs to be refactored to actually work with the NZGMDB
"""

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
        print(f"SNR filename: {filename}")

        # getting only the snr columns
        snr = snr_all_cols[["snr_000", "snr_090", "snr_ver"]]

        # Smoothing data using pandas rolling mean function
        # snr_000 = (
        #     snr["snr_000"]
        #     .rolling(window=config.get_value("max_freq_Hz"), center=True, min_periods=1)
        #     .mean()
        # )
        # snr_090 = snr["snr_090"].rolling(window=5, center=True, min_periods=1).mean()
        # snr_ver = snr["snr_ver"].rolling(window=5, center=True, min_periods=1).mean()

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
            (snr_all_cols["frequency"] >= config.get_value("min_freq_Hz"))
            & (snr_all_cols["frequency"] <= config.get_value("max_freq_Hz"))
        ]

        num_valid_points_in_interval = np.sum(
            snr_smooth_freq_interval_for_screening
            > [
                config.get_value("snr_thresh_horiz"),
                config.get_value("snr_thresh_horiz"),
                config.get_value("snr_thresh_vert"),
            ],
            axis=0,
        )

        if not (
            num_valid_points_in_interval > config.get_value("min_points_above_thresh")
        ).all():
            ### To do: add to a list of skipped records with the reason such as "failed initial checks"
            continue

        print("bp")

        id_4hz = freq > 4.0
        freq_4hz = freq[id_4hz]

        # Compute fmax for each component
        loc_000 = np.where(snr_000[id_4hz] < 3)[0]
        if len(loc_000) != 0:
            fmax_000 = min(freq_4hz[loc_000[0]], fny)
        else:
            fmax_000 = min(freq_4hz[-1], fny)

        loc_090 = np.where(snr_090[id_4hz] < 3)[0]
        if len(loc_090) != 0:
            fmax_090 = min(freq_4hz[loc_090[0]], fny)
        else:
            fmax_090 = min(freq_4hz[-1], fny)

        loc_ver = np.where(snr_ver[id_4hz] < 3)[0]
        if len(loc_ver) != 0:
            fmax_ver = min(freq_4hz[loc_ver[0]], fny)
        else:
            fmax_ver = min(freq_4hz[-1], fny)

        # Storing computed fmax values for different components
        fmax_000_list.append(fmax_000)
        fmax_090_list.append(fmax_090)
        fmax_ver_list.append(fmax_ver)
        ids.append(ev_sta_chan)

    return fmax_000_list, fmax_090_list, fmax_ver_list, ids


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
