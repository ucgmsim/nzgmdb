from pathlib import Path

import mseedlib
import numpy as np
import pandas as pd
from obspy.core import Stream, Trace, UTCDateTime

from IM_calculation.IM import read_waveform
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
    print(f"Reading mseed file {mseed_file}")
    # Read the mseed file
    mseed = read_mseed_to_stream(mseed_file)

    if len(mseed) != 3:
        raise custom_errors.All3ComponentsNotPresentError(
            f"All 3 components are not present in the mseed file {mseed_file}"
        )

    # Process the data if needed
    if pre_process:
        print(f"Pre-processing data from {mseed_file}")
        mseed = waveform_manipulation.initial_preprocessing(mseed)

    # Stack the data
    print(f"Stacking data from {mseed_file}")
    try:
        data = np.stack([tr.data for tr in mseed], axis=1)
        data = data.astype(np.float64)
    except ValueError:
        print(f"Error reading data from {mseed_file}")
        raise custom_errors.InvalidTraceLengthError(
            f"Error reading data from {mseed_file}"
        )

    print(f"Creating waveform object from {mseed_file}")

    # Create the waveform object
    waveform = read_waveform.create_waveform_from_data(
        data, NT=len(mseed[0].data), DT=mseed[0].stats.delta
    )

    print(f"Waveform object created from {mseed_file}")

    return waveform


def create_waveform_from_processed(
    ffp_000: Path,
    ffp_090: Path,
    ffp_ver: Path,
    delta: float = None,
):
    """
    Create a waveform object from processed data using the 3 component files

    Parameters
    ----------
    ffp_000 : Path
        Path to the 000 component file
    ffp_090 : Path
        Path to the 090 component file
    ffp_ver : Path
        Path to the vertical component file
    delta : float
        The time step between each data point

    Returns
    -------
    Waveform
        The waveform object created from the data
    """
    # Load all components
    comp_000 = pd.read_csv(ffp_000, sep=r"\s+", header=None, skiprows=2).values.ravel()
    comp_090 = pd.read_csv(ffp_090, sep=r"\s+", header=None, skiprows=2).values.ravel()
    comp_ver = pd.read_csv(ffp_ver, sep=r"\s+", header=None, skiprows=2).values.ravel()

    if delta is None:
        # Get the DT value from 2nd row 2nd value
        delta = pd.read_csv(ffp_000, sep=r"\s+", header=None, nrows=2, skiprows=1).iloc[
            0, 1
        ]

    # Remove NaN values
    comp_000 = comp_000[~np.isnan(comp_000)]
    comp_090 = comp_090[~np.isnan(comp_090)]
    comp_ver = comp_ver[~np.isnan(comp_ver)]

    # Form the waveform
    waveform = read_waveform.create_waveform_from_data(
        np.stack((comp_000, comp_090, comp_ver), axis=1), NT=len(comp_000), DT=delta
    )

    return waveform
