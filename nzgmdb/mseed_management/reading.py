from pathlib import Path
import concurrent.futures

import mseedlib
import numpy as np
import obspy
import pandas as pd

from IM_calculation.IM import read_waveform
from nzgmdb.data_processing import waveform_manipulation
from nzgmdb.management import custom_errors


def read_mseed_with_timeout(mseed_file: Path, timeout: int = 20, max_retries: int = 3):
    def read_mseed(file):
        return obspy.read(str(file))

    for attempt in range(max_retries):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(read_mseed, mseed_file)
            try:
                mseed = future.result(timeout=timeout)
                return mseed
            except concurrent.futures.TimeoutError:
                print(f"Attempt {attempt + 1} timed out. Retrying...")
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                break
    raise Exception(
        f"Failed to read mseed file {mseed_file} after {max_retries} attempts"
    )


class Trace:
    def __init__(self, data, channel_id):
        self.data = data
        self.channel_id = channel_id


class Mseed:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.start_time = None
        self.end_time = None
        self.dt = None
        self.event_id = None
        self.station = None
        self.location = None
        self.channel = None
        self.traces = []

        # Read the file and extract information
        nptype = {"i": np.int32, "f": np.float32, "d": np.float64, "t": np.char}
        mstl = mseedlib.MSTraceList()
        mstl.read_file(str(file_path), unpack_data=False, record_list=True)

        for traceid in mstl.traceids():
            for segment in traceid.segments():
                # Fetch estimated sample size and type
                (sample_size, sample_type) = segment.sample_size_type
                dtype = nptype[sample_type]
                # Allocate NumPy array for data samples
                data_samples = np.zeros(segment.samplecnt, dtype=dtype)
                # Unpack data samples into allocated NumPy array
                segment.unpack_recordlist(
                    buffer_pointer=np.ctypeslib.as_ctypes(data_samples),
                    buffer_bytes=data_samples.nbytes,
                )
                # Get the component
                comp = mseedlib.sourceid2nslc(traceid.sourceid)[-1][-1]
                # Add the trace
                self.traces.append(Trace(data_samples, comp))


def read_mseed(mseed_file: Path):
    """
    Reads a mseed file using mseedlib to avoid potential process stalling

    Parameters
    ----------
    mseed_file : Path
        The mseed file to read
    """
    nptype = {"i": np.int32, "f": np.float32, "d": np.float64, "t": np.char}
    mstl = mseedlib.MSTraceList()
    mstl.read_file(str(mseed_file), unpack_data=False, record_list=True)

    for traceid in mstl.traceids():
        for segment in traceid.segments():
            # Fetch estimated sample size and type
            (sample_size, sample_type) = segment.sample_size_type
            dtype = nptype[sample_type]
            # Allocate NumPy array for data samples
            data_samples = np.zeros(segment.samplecnt, dtype=dtype)
            # Unpack data samples into allocated NumPy array
            segment.unpack_recordlist(
                buffer_pointer=np.ctypeslib.as_ctypes(data_samples),
                buffer_bytes=data_samples.nbytes,
            )
            # Create a dictionary for the trace with basic metadata
            comp = mseedlib.sourceid2nslc(traceid.sourceid)[-1][-1]
            output_dict[comp_mapping[comp]] = {
                "start_time": segment.starttime_str(),
                "end_time": segment.endtime_str(),
                "dt": 1 / segment.samprate,
                "data_samples": data_samples.tolist(),
            }

    return mstl


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
    # mseed = obspy.read(str(mseed_file))
    try:
        # mseed = read_mseed_with_timeout(mseed_file)
        mseed = read_mseed(mseed_file)
    except Exception as e:
        raise custom_errors.All3ComponentsNotPresentError(
            f"Error reading mseed file {mseed_file} with error: {e}"
        )

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
        data, NT=mseed[0].stats.npts, DT=mseed[0].stats.delta
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
