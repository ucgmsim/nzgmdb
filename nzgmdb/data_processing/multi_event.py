import numpy as np
import pandas as pd
from obspy.core.stream import Stream, Trace
from obspy.signal.trigger import recursive_sta_lta, trigger_onset

from nzgmdb.data_processing import waveform_manipulation
from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors


def sync_event_from_stream(
    stream: Stream,
    extraction_df: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    """
    Determine trace start/end times from an ObsPy Stream and whether the
    station has a catalog pick inside that window.

    Parameters
    ----------
    stream : obspy.Stream
        Input stream (multi-component). A Z component is preferred, otherwise
        the first trace is used.
    extraction_df : pandas.DataFrame
        DataFrame containing catalog picks with a `ptime_est` column for the same site
        for other events.

    Returns
    -------
    start_time : pandas.Timestamp
        UTC start time of the selected trace (or `pd.NaT` on failure).
    end_time : pandas.Timestamp
        UTC end time of the selected trace (or `pd.NaT` on failure).
    sync_event : bool
        True if there is at least one pick inside the trace window, else False.
    """
    trace = stream[0]

    # Convert start/end to pandas UTC timestamps
    start_time = pd.to_datetime(trace.stats.starttime.datetime, utc=True)
    end_time = pd.to_datetime(trace.stats.endtime.datetime, utc=True)

    t = pd.to_datetime(extraction_df["ptime_est"], utc=True)

    # Check for any pick inside the window (inclusive)
    inside = t.between(start_time, end_time)
    sync_event = inside.any()

    return start_time, end_time, sync_event


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
    float
        Weighted multi-trigger score based on STA/LTA triggers, or np.nan on failure to process.
    """

    # Ensure reproducible component order
    try:
        stream = waveform_manipulation.initial_preprocessing(
            stream, apply_zero_padding=False
        )
    except (
        custom_errors.InventoryNotFoundError
        or custom_errors.SensitivityRemovalError
        or custom_errors.RotationError
    ):
        return np.nan

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

    return weighted_score


def compute_multi_event_scores(stream: Stream, extraction_table: pd.DataFrame):
    """
    Compute multi-event scores for a given ObsPy Stream and extraction table.

    Parameters
    ----------
    stream : obspy.Stream
        Input multi-component stream.
    extraction_table : pandas.DataFrame
        DataFrame containing catalog picks with a `ptime_est` column for the same site
        for other events.

    Returns
    -------
    start_time : pandas.Timestamp
        UTC start time of the selected trace (or `pd.NaT` on failure).
    end_time : pandas.Timestamp
        UTC end time of the selected trace (or `pd.NaT` on failure).
    stalat_score : bool
        Multi-event score based on STA/LTA triggers.
    sync_event : bool
        True if there is at least one pick inside the trace window, else False.
    """
    start_time, end_time, sync_event = sync_event_from_stream(stream, extraction_table)

    stalat_score = stalta_for_stream(stream)

    return start_time, end_time, stalat_score, sync_event
