"""
This module contains the functions for determining the clip probability of a given mseed file
"""

import numpy as np
from gmprocess.waveform_processing.clipping.clipping_ann import clipNet
from gmprocess.waveform_processing.clipping.histogram import Histogram
from gmprocess.waveform_processing.clipping.max_amp import MaxAmp
from gmprocess.waveform_processing.clipping.ping import Ping
from obspy import Stream

from nzgmdb.management import config as cfg


def get_clip_probability(event_mag: float, dist: float, mseed: Stream) -> float:
    """
    Calculate the clip probability based on the mseed inputs

    Parameters
    ----------
    event_mag : float
        The magnitude of the event
    dist : float
        The distance of the event to the station
    mseed : Stream
        The mseed Stream object

    Returns
    -------
    float
        The clip probability from ClipNet
    """
    # Get the config values
    config = cfg.Config()
    mag_clip_low = config.get_value("mag_clip_low")
    mag_clip_high = config.get_value("mag_clip_high")
    dist_clip_low = config.get_value("dist_clip_low")
    dist_clip_high = config.get_value("dist_clip_high")

    # Ensure numeric inputs
    try:
        event_mag = float(event_mag)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"event_mag must be a number, got {event_mag!r}") from exc

    try:
        dist = float(dist)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"dist must be a number, got {dist!r}") from exc

    # Clip the event_mag and dist values
    event_mag = np.clip(event_mag, mag_clip_low, mag_clip_high)
    dist = np.clip(dist, dist_clip_low, dist_clip_high)

    # Get different methods for clipping
    max_amp_method = MaxAmp(mseed)
    hist_method = Histogram(mseed)
    ping_method = Ping(mseed)

    # Define the inputs for the clipNet
    inputs = [
        event_mag,
        dist,
        max_amp_method.is_clipped,
        hist_method.is_clipped,
        ping_method.is_clipped,
    ]
    # Get the clip probability
    clip_nnet = clipNet()
    return clip_nnet.evaluate(inputs)[0][0]


def get_jerk(mseed: Stream) -> bool:
    """
    Calculate if the mseed has jerk that exceeds the threshold for any trace
    that exceeds median_multiplier the median jerk.

    Parameters
    ----------
    mseed : Stream
        The mseed Stream object

    Returns
    -------
    bool
        True if the mseed has jerk in any trace, False otherwise
    """
    # Get the config values
    config = cfg.Config()
    point_thresh = config.get_value("point_thresh")
    median_multiplier = config.get_value("median_multiplier")

    # Check for jerk in each trace
    for trace in mseed:
        temp_tr = trace.copy()
        temp_tr.differentiate()
        temp_tr.differentiate()
        abs_diff = np.abs(temp_tr.data)
        median_multiplied = median_multiplier * np.median(abs_diff)
        (i_jerk,) = np.where(abs_diff >= median_multiplied)
        num_outliers = len(i_jerk)
        if num_outliers > point_thresh:
            return True
    return False
