from pathlib import Path

import mseedlib
import numpy as np
from obspy.core import Stream, Trace, UTCDateTime

from nzgmdb.data_processing import waveform_manipulation
from nzgmdb.management import custom_errors


def read_mseed_to_stream(file_path: Path):
    """
    Read a MiniSEED file using mseedlib and convert it to an ObsPy Stream object.

    Parameters
    ----------
    file_path : Path
        Path to the MiniSEED file

    Returns
    -------
    Stream
        ObsPy Stream object containing the data from the MiniSEED file
    """
    stream = Stream()
    nptype = {"i": np.int32, "f": np.float32, "d": np.float64, "t": np.char}
    mstl = mseedlib.MSTraceList()
    mstl.read_file(str(file_path), unpack_data=False, record_list=True)

    for traceid in mstl.traceids():
        for segment in traceid.segments():
            # Determine data type and allocate array
            (sample_size, sample_type) = segment.sample_size_type
            dtype = nptype[sample_type]
            data_samples = np.zeros(segment.samplecnt, dtype=dtype)

            # Unpack data samples
            segment.unpack_recordlist(
                buffer_pointer=np.ctypeslib.as_ctypes(data_samples),
                buffer_bytes=data_samples.nbytes,
            )

            # Get metadata
            sourceid = traceid.sourceid.split("FDSN:")[1]
            parts = sourceid.split("_")
            if len(parts) > 4:
                network, station, location, *channel = parts
                channel = "".join(channel)
            else:
                network, station, location, channel = parts
            start_time = UTCDateTime(segment.starttime_seconds)
            sampling_rate = segment.samprate

            # Create ObsPy Trace and add to Stream
            trace = Trace(data=data_samples)
            trace.stats.network = network
            trace.stats.station = station
            trace.stats.location = location
            trace.stats.channel = channel
            trace.stats.starttime = start_time
            trace.stats.sampling_rate = sampling_rate
            stream.append(trace)

    return stream


def create_waveform_from_mseed(
    mseed_file: Path,
    pre_process: bool = False,
    apply_zero_padding: bool = True,
    apply_taper: bool = True,
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
    apply_zero_padding : bool (optional)
        Whether to apply zero padding to the waveform, by default True (Only used if pre_process is True)
    apply_taper : bool (optional)
        Whether to apply a taper to the waveform, by default True (Only used if pre_process is True)

    Returns
    -------
    np.ndarray
        The waveform data in the shape (1, n_samples, 3)

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
    mseed = read_mseed_to_stream(mseed_file)

    if len(mseed) != 3:
        raise custom_errors.All3ComponentsNotPresentError(
            f"All 3 components are not present in the mseed file {mseed_file}"
        )

    # Process the data if needed
    if pre_process:
        mseed = waveform_manipulation.initial_preprocessing(
            mseed, apply_taper, apply_zero_padding
        )

    # Stack the data
    try:
        data = np.stack([tr.data for tr in mseed], axis=1)
        data = data.astype(np.float32)
        # Reshape the waveform to have the correct shape for the IM calculation
        reshaped_waveform = data[np.newaxis, :, :]
    except ValueError:
        raise custom_errors.InvalidTraceLengthError(
            f"Error reading data from {mseed_file}"
        )

    return reshaped_waveform
