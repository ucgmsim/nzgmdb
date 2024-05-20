from pathlib import Path

import numpy as np
import obspy

from IM_calculation.IM import read_waveform
from nzgmdb.management import custom_errors
from nzgmdb.data_processing import waveform_manipulation


def create_waveform_from_mseed(
    mseed_file: Path,
    pre_process: bool = False,
):
    """
    Create a waveform object from a mseed file
    Can perform some simple processing such as detrending and removing sensitivity if possible

    Parameters
    ----------
    mseed_file : Path
        Path to the mseed file
    pre_process : bool (optional)
        Whether to do some small processing such as detrending and removing sensitivity
        (Can however fail if the sensitivity can't be removed and raise errors), by default False

    Returns
    -------
    Waveform
        The waveform object created from the mseed file

    Raises
    ------
    InventoryNotFoundError
        If no inventory information is found for the station and location pair
    SensitivityRemovalError
        If the sensitivity removal fails
    All3ComponentsNotPresentError
        If all 3 components are not present in the mseed file
    """
    # Read the mseed file
    mseed = obspy.read(str(mseed_file))

    if len(mseed) != 3:
        raise custom_errors.All3ComponentsNotPresentError(
            f"All 3 components are not present in the mseed file {mseed_file}"
        )

    # Process the data if needed
    if pre_process:
        mseed = waveform_manipulation.initial_preprocessing(mseed)

    # Stack the data
    data = np.stack([tr.data for tr in mseed], axis=1)
    data = data.astype(np.float64)

    # Create the waveform object
    waveform = read_waveform.create_waveform_from_data(
        data, NT=mseed[0].stats.npts, DT=mseed[0].stats.delta
    )

    return waveform
