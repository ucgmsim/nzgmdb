from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read
from obspy.core.stream import Stream, Trace
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

from nzgmdb.data_processing import waveform_manipulation
from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure


def stalta_triggers(tr: Trace):
    """
    Compute STA/LTA characteristic function and extract cleaned triggers.

    Parameters
    ----------
    tr : obspy.Trace
        Single-component trace to analyze.

    Returns
    -------
    flag : int
        Binary indicator of multiple cleaned triggers. Returns 1 if more than
        one cleaned trigger is found, otherwise 0.
    """
    # Get the config parameters
    config = cfg.Config()
    sta_s = config.get_value("sta_window_s")
    lta_s = config.get_value("lta_window_s")
    on_thr = config.get_value("on_threshold")
    off_thr = config.get_value("off_threshold")
    min_dur_s = config.get_value("min_duration_s")
    min_gap_s = config.get_value("min_gap_s")
    edge_skip_s = config.get_value("edge_skip_s")

    sr = float(tr.stats.sampling_rate)

    nsta = max(1, int(sr * sta_s))
    nlta = max(nsta + 1, int(sr * lta_s))

    cft = recursive_sta_lta(tr.data.astype(np.float64), nsta, nlta)

    # Initial raw trigger windows
    on_off = trigger_onset(cft, on_thr, off_thr)

    # Remove triggers near trace edges
    start_cut = int(edge_skip_s * sr)
    end_cut = len(tr.data) - int(edge_skip_s * sr)

    on_off = np.array(
        [win for win in on_off if (win[0] >= start_cut and win[1] <= end_cut)],
        dtype=int,
    )

    # Enforce minimum trigger duration
    if min_dur_s > 0 and len(on_off) > 0:
        min_len = int(sr * min_dur_s)
        on_off = np.array(
            [win for win in on_off if (win[1] - win[0]) >= min_len], dtype=int
        )

    # Merge triggers separated by small gaps
    if len(on_off) > 1 and min_gap_s > 0:
        merged = []
        gap_samples = int(sr * min_gap_s)

        current = on_off[0].tolist()
        for nxt in on_off[1:]:
            # If the next trigger starts soon after the previous ends → merge
            if nxt[0] - current[1] <= gap_samples:
                current[1] = max(current[1], nxt[1])
            else:
                merged.append(current)
                current = nxt.tolist()

        merged.append(current)
        on_off = np.array(merged, dtype=int)

    # Count and flag
    count = len(on_off)
    flag = int(count > 1)

    return flag


def stalta_for_stream(stream: Stream):
    """
    Run STA/LTA detection for a 3-component stream (H1, H2, Z).
    Returns weighted multi-trigger score.

    Parameters:
    -----------
    stream : obspy.Stream
        Input 3-component stream.

    Returns:
    --------
    bool
        Multi-trigger score (True if weighted score > 0.5, else False).
    """

    # Ensure reproducible component order
    stream = waveform_manipulation.initial_preprocessing(
        stream, apply_zero_padding=False
    )
    stream.sort(keys=["channel"])
    tr_H1, tr_H2, tr_Z = stream[0], stream[1], stream[2]

    # Run detector on each component
    flag_H1 = stalta_triggers(tr_H1)
    flag_H2 = stalta_triggers(tr_H2)
    flag_Z = stalta_triggers(tr_Z)

    config = cfg.Config()
    weights = config.get_value("weights")

    # Weighted multi-trigger score
    weighted_score = (
        weights["h1"] * flag_H1 + weights["h2"] * flag_H2 + weights["z"] * flag_Z
    )

    multi_event_score = True if weighted_score > 0.5 else False

    return multi_event_score


def compute_stalta_scores(df: pd.DataFrame, main_dir: Path) -> pd.DataFrame:
    """
    For each row in the dataframe get the mseed file path and compute the
    STA/LTA multi-trigger score. Add the score as a new column to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing event information and mseed file paths.
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with an additional 'multi_event' column containing
        the STA/LTA multi-trigger score of True/False.
    """
    multi_event_values = []
    # Ensure datetime column is in datetime format
    df["datetime"] = pd.to_datetime(df["datetime"])

    for _, row in df.iterrows():
        # Get the mseed directory
        year = row["datetime"].year
        evid = row["evid"]
        mseed_dir = file_structure.get_mseed_dir(main_dir, year, evid)

        # Get the mseed file path
        record_id = row["record_id"]
        mseed_file = mseed_dir / f"{record_id}.mseed"

        # Read the mseed file and compute the multi-trigger score
        stream = read(mseed_file)
        multi_event_values.append(stalta_for_stream(stream))

    return df
