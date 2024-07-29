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
    """
    # Get the config values
    config = cfg.Config()
    mag_clip_low = config.get_value("mag_clip_low")
    mag_clip_high = config.get_value("mag_clip_high")
    dist_clip_low = config.get_value("dist_clip_low")
    dist_clip_high = config.get_value("dist_clip_high")

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
