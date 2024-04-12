# fmax calculation

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable
import multiprocessing as mp
import functools

# File inputs
metadata_df = pd.read_csv('/home/joel/local/gmdb/new_data/metadata.csv')
snr_dir = Path("/home/joel/local/snr/snr_new_p_wave_v2_full")
snr_filenames = snr_dir.glob("*_snr_fas.csv")

# Add the ev_sta_chan column to the metadata
metadata_df["ev_sta_chan"] = metadata_df['evid'] + '_' + metadata_df['sta'] + '_' + metadata_df['chan']


def find_fmaxs(filenames: Iterable[Path], metadata: pd.DataFrame):
    # Create empty lists to store fmax values
    fmax_000_list = []
    fmax_090_list = []
    fmax_ver_list = []
    ids = []

    # Iterate over the filenames
    for idx, filename in enumerate(filenames):
        ev_sta_chan = str(filename.stem).replace('_snr_fas', '')

        # Get Delta from the metadata
        current_row = metadata.iloc[np.where(metadata["ev_sta_chan"] == ev_sta_chan)[0], :]
        fny = 1 / current_row['delta'].iloc[0] * 0.5 * 0.8

        # Read CSV file using pandas
        snr = pd.read_csv(filename)
        freq = snr['frequency']

        # Smoothing data using pandas rolling mean function
        snr_000 = snr['snr_000'].rolling(window=5, center=True, min_periods=1).mean()
        snr_090 = snr['snr_090'].rolling(window=5, center=True, min_periods=1).mean()
        snr_ver = snr['snr_ver'].rolling(window=5, center=True, min_periods=1).mean()

        # Initial screening: at least n_min freq points btw 0.5-10 Hz with SNR>5
        n_min = 5
        if ((snr_000.iloc[33:77] > 5).sum() < n_min) or ((snr_090.iloc[33:77] > 5).sum() < n_min) or ((snr_ver.iloc[33:77] > 3).sum() < n_min):
            continue

        id_4hz = freq > 4.0
        freq_4hz = freq[id_4hz].to_numpy()

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


n_procs = 8

# Split the filenames into n_procs chunks
snr_filenames = np.array_split(list(snr_filenames), n_procs)

with mp.Pool(n_procs) as p:
    results = p.map(
        functools.partial(
            find_fmaxs,
            metadata=metadata_df,
        ),
        snr_filenames,
    )

# Combine the results into the 4 different lists
fmax_000_list = np.concatenate([result[0] for result in results])
fmax_090_list = np.concatenate([result[1] for result in results])
fmax_ver_list = np.concatenate([result[2] for result in results])
ids = np.concatenate([result[3] for result in results])

# Create fmax csv
fmax = pd.DataFrame({'ev_sta': ids, 'fmax_000': fmax_000_list, 'fmax_090': fmax_090_list, 'fmax_ver': fmax_ver_list})
fmax.to_csv(snr_dir.parent / 'fmax.csv', index=False)

