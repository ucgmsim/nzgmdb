import http.client
import warnings
from collections.abc import Iterable
from pathlib import Path

import mseedlib
import numpy as np
import pandas as pd
from obspy import Stream
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.event import Origin
from obspy.geodetics import kilometers2degrees
from obspy.io.mseed import InternalMSEEDError, ObsPyMSEEDFilesizeTooSmallError
from obspy.taup import TauPyModel

from empirical.util import classdef, openquake_wrapper_vectorized, z_model_calculations
from nzgmdb.management import config as cfg


def get_waveforms(
    preferred_origin: Origin,
    client: FDSN_Client,
    net: str,
    sta: str,
    mag: float,
    rrup: float,
    r_epi: float,
    vs30: float = None,
):
    """
    Get the waveforms for a given event and station
    Calculate the start and end times for the waveform based on the P and S arrival times

    Parameters
    ----------
    preferred_origin : Origin
        The preferred origin object from the event catalogue
    client : FDSN_Client
        The FDSN client to use to get the waveforms
    net : str
        The network code
    sta : str
        The station
    mag : float
        The magnitude of the event
    rrup : float
        The closest distance to the event
    r_epi : float
        The epicentral distance to the event
    vs30 : float, optional
        The Vs30 value for the station, by default sets to config value

    Returns
    -------
    st : Union[Stream, None]
        The stream object containing the waveform data or None if no data is found
    """
    config = cfg.Config()
    vs30 = config.get_value("vs30") if vs30 is None else vs30
    rake = 90  # Assume strike-slip for now
    z1p0 = z_model_calculations.chiou_young_08_calc_z1p0(vs30)
    # Predict significant duration time from Afshari and Stewart (2016)
    input_df = pd.DataFrame(
        {
            "mag": [mag],
            "rake": [rake],
            "rrup": [rrup],
            "vs30": [vs30],
            "z1pt0": [z1p0],
        }
    )
    result_df = openquake_wrapper_vectorized.oq_run(
        classdef.GMM.AS_16,
        classdef.TectType.ACTIVE_SHALLOW,
        input_df,
        "Ds595",
    )
    ds = np.exp(result_df["Ds595_mean"].values[0])

    deg = kilometers2degrees(r_epi)

    model = TauPyModel(model="iasp91")

    # Estimate arrival times for P and S phases
    p_arrivals = model.get_travel_times(
        source_depth_in_km=preferred_origin.depth / 1000,
        distance_in_degree=deg,
        phase_list=["ttp"],
    )
    s_arrivals = model.get_travel_times(
        source_depth_in_km=preferred_origin.depth / 1000,
        distance_in_degree=deg,
        phase_list=["tts"],
    )
    ptime_est = (
        preferred_origin.time + p_arrivals[0].time
    )  # Estimated earliest P arrival time from taup
    stime_est = preferred_origin.time + s_arrivals[0].time

    # Find the start and end times for the mseed data
    min_time_difference = config.get_value("min_time_difference")
    ds_multiplier = config.get_value("ds_multiplier")
    start_time = ptime_est - min_time_difference
    end_time = stime_est + (
        min_time_difference
        if stime_est + ds * ds_multiplier - ptime_est < min_time_difference
        else ds * ds_multiplier
    )

    channel_codes = ",".join(config.get_value("channel_codes"))
    # Get the waveforms with multiple retries when IncompleteReadError occurs
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                st = client.get_waveforms(
                    net,
                    sta,
                    "*",
                    channel_codes,
                    start_time,
                    end_time,
                    attach_response=True,
                )
            break
        except FDSNNoDataException:
            return None
        except ObsPyMSEEDFilesizeTooSmallError:
            return None
        except (http.client.IncompleteRead, InternalMSEEDError):
            if attempt < max_retries - 1:  # i.e. not the last attempt
                continue  # try again
            else:
                return None
        except Exception as e:
            print(f"Error getting waveforms for {net}.{sta}")
            print(e)
            return None
    print(f"Got Waveform for station {sta}")
    return st


def split_stream_into_mseeds(st: Stream, unique_channels: Iterable):
    """
    Split the stream object into multiple mseed files based on the unique channel and location

    Parameters
    ----------
    st : Stream
        The stream object containing the waveform data for every channel and location
    unique_channels : Iterable
        An Iterable of tuples containing the unique channel and location for each mseed file created
        [(channel, location), ...]

    Returns
    -------
    mseeds : list
        A list of stream objects containing the waveform data for each mseed file created
    """
    mseeds = []
    for chan, loc in unique_channels:
        # Each unique channel and location pair is a new mseed file
        st_new = st.select(location=loc, channel=f"{chan}?")

        if len(st_new) > 3:
            # Check if all the sample rates are the same
            samples = [tr.stats.sampling_rate for tr in st_new]
            if len(set(samples)) > 1:
                # If they are different take the highest and resample with the others using interpolation
                st_new = st_new.select(sampling_rate=max(samples))
                st_new.merge(fill_value="interpolate")

        # Check the final length of the traces
        if len(st_new) != 3:
            continue

        # Ensure traces all have the same length
        starttime_trim = max([tr.stats.starttime for tr in st_new])
        endtime_trim = min([tr.stats.endtime for tr in st_new])
        # Check that the start time is before the end time
        if starttime_trim > endtime_trim:
            continue
        st_new.trim(starttime_trim, endtime_trim)

        mseeds.append(st_new)

    return mseeds


def write_stream_to_mseed(stream: Stream, output_file: Path):
    """
    Write an ObsPy Stream object to a MiniSEED file using mseedlib.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        The Stream object to write to MiniSEED
    output_file : Path
        The path to the output MiniSEED file

    Raises
    ------
    ValueError
        If the sample type of the trace data is not supported
    """
    mstl = mseedlib.MSTraceList()

    for trace in stream:
        # Construct FDSN source ID
        sourceid = f"FDSN:{trace.stats.network}_{trace.stats.station}_{trace.stats.location}_{trace.stats.channel}"

        # Convert start time to nanoseconds
        start_time = mseedlib.timestr2nstime(
            trace.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        )

        # Determine sample type
        if trace.data.dtype == np.int32:
            sample_type = "i"
        elif trace.data.dtype == np.float32:
            sample_type = "f"
        elif trace.data.dtype == np.float64:
            sample_type = "d"
        else:
            raise ValueError(f"Unsupported sample type: {trace.data.dtype}")

        # Add trace data to MSTraceList
        mstl.add_data(
            sourceid=sourceid,
            data_samples=trace.data.tolist(),
            sample_type=sample_type,
            sample_rate=trace.stats.sampling_rate,
            start_time=start_time,
        )

    # Record handler to write MiniSEED records to file
    def record_handler(record: bytes, handler_data: dict):
        """
        Write MiniSEED record to file handler.

        Parameters
        ----------
        record : bytes
            The MiniSEED record to write
        handler_data : dict
            Dictionary containing the file handler to write
        """
        handler_data["fh"].write(record)

    # Write to MiniSEED file
    with open(output_file, "wb") as file_handle:
        mstl.pack(record_handler, {"fh": file_handle}, flush_data=True)


def write_mseed(mseed: Stream, event_id: str, station: str, output_directory: Path):
    """
    Write the mseed files to the output directory

    Parameters
    ----------
    mseed : Stream
        The stream object containing the waveform data
    event_id : str
        The event id which is used in the filename
    station : str
        The station code which is used in the filename
    output_directory : Path
        The directory to save the mseed files
    """
    # Get the channel and location from the first trace
    channel = mseed[0].stats.channel[:2]
    location = mseed[0].stats.location

    # Create the filename and add it to the output directory
    filename = f"{event_id}_{station}_{channel}_{location}.mseed"
    mseed_ffp = output_directory / filename
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write the mseed file
    write_stream_to_mseed(mseed, mseed_ffp)
