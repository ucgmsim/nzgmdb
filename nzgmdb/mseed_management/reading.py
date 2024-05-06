import obspy
import numpy as np
from pathlib import Path

from IM_calculation.IM import read_waveform
from nzgmdb.data_processing import waveform_manipulation


def create_waveform_from_mseed(
        mseed_file: Path,
        process: bool = False,
):
    """
    Create a waveform object from a mseed file
    Can perform some simple processing such as detrending and removing sensitivity if possible

    Parameters
    ----------
    mseed_file : Path
        Path to the mseed file
    process : bool (optional)
        Whether to do some small processing such as detrending and removing sensitivity
        (Can however fail if the sensitivity can't be removed and will return None)

    Returns
    -------
    Waveform, None
        The waveform object created from the mseed file
        or None if processing failed
    """
    # Read the mseed file
    mseed = obspy.read(str(mseed_file))

    # Process the data if needed
    if process:
        mseed = waveform_manipulation.basic_manipulation(mseed)
        if mseed is None:
            return None

    # Stack the data
    data = np.stack([tr.data for tr in mseed], axis=1)
    data = data.astype(np.float64)

    # Create the waveform object
    waveform = read_waveform.create_waveform_from_data(
        data, NT=mseed[0].stats.npts, DT=mseed[0].stats.delta
    )

    return waveform
