import obspy
import numpy as np
from pathlib import Path

from IM_calculation.IM import read_waveform


def create_waveform_from_mseed(
        mseed_file: Path,
):
    """
    Create a waveform object from a mseed file

    Parameters
    ----------
    mseed_file : Path
        Path to the mseed file
    """
    # Read the mseed file
    mseed = obspy.read(str(mseed_file))

    # Get the data
    data = np.stack([tr.data for tr in mseed], axis=1)
    data = data.astype(np.float64)

    # Create the waveform object
    waveform = read_waveform.create_waveform_from_data(
        data, NT=mseed[0].stats.npts, DT=mseed[0].stats.delta
    )

    return waveform


create_waveform_from_mseed(Path("/home/joel/local/gmdb/testing_folder/waveforms/2021/2021p001797/mseed/2021p001797_WANS_HN_20.mseed"))