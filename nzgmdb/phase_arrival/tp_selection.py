"""
Functions for getting specific times / indexes for p / s waves from the phase arrival table
"""

from pathlib import Path

import pytz
import pandas as pd
from obspy.core import Stats


def get_tp_from_phase_table(phase_table_path: Path, mseed_stats: Stats, event_id: str):
    """
    Get the index of the p-wave arrival from the phase arrival table
    Using the event id and the mseed file to gather metadata

    Parameters
    ----------
    phase_table_path : Path
        Path to the phase arrival table
    mseed_stats : Stats
        Stats object from the mseed file
    event_id : str
        Event id of the event

    Returns
    -------
    tp: Index of the p-wave arrival for the waveform array,
        None if no p-wave arrival could be found
    """
    phase_arrival_table = pd.read_csv(phase_table_path, low_memory=False)

    # Getting the appropriate row from the phase arrival table
    event_df = phase_arrival_table.loc[
        (phase_arrival_table["evid"] == event_id)
        & (phase_arrival_table["sta"] == mseed_stats.station)
    ]
    # If there is no data for the event / station pair, return None
    if len(event_df) == 0:
        return None

    # Select the correct phase (Needs to start with a P)
    phase_row = event_df.loc[event_df["phase"].str.upper().str.startswith("P")]

    # If there is no P phase, return None
    if len(phase_row) == 0:
        return None

    # Get the time of the P phase
    tp_time = pd.Timestamp(phase_row["datetime"].values[0], tz=pytz.UTC)

    # Datetime conversions to timestamps for comparisions with and without UTC
    dt_start = mseed_stats.starttime
    dt_end = mseed_stats.endtime
    dt_start_UTC = pd.Timestamp(
        year=dt_start.year,
        month=dt_start.month,
        day=dt_start.day,
        hour=dt_start.hour,
        minute=dt_start.minute,
        second=dt_start.second,
        microsecond=dt_start.microsecond,
        tz=pytz.UTC,
    )
    dt_end_UTC = pd.Timestamp(
        year=dt_end.year,
        month=dt_end.month,
        day=dt_end.day,
        hour=dt_end.hour,
        minute=dt_end.minute,
        second=dt_end.second,
        microsecond=dt_end.microsecond,
        tz=pytz.UTC,
    )

    # Calculate the time difference between the start and end times
    time_diff = dt_end_UTC - dt_start_UTC

    # Calculate the time difference between each point
    interval = time_diff / (mseed_stats.npts - 1)

    # Calculate the position of the tp time within the time range
    desired_position = (tp_time - dt_start_UTC) / interval

    # Round the desired position to the nearest whole number
    tp = round(desired_position)

    # Ensure the tp is within the range of the waveform
    if tp > mseed_stats.npts or tp < 0:
        return None
    else:
        return tp
