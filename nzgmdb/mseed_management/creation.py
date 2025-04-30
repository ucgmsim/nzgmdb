"""
This module contains functions for creating mseed files from the waveform data from the FDSN client
"""

import http
import http.client
import time
import warnings
from collections.abc import Iterable
from pathlib import Path

import mseedlib
import numpy as np
import pandas as pd
from obspy import Stream
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import (
    FDSNNoDataException,
    FDSNServiceUnavailableException,
)
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
    only_record_ids: list[str] = None,
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
    only_record_ids : list[str], optional
        A list of record ids to get the waveforms for, by default None

    Returns
    -------
    Union[Stream, None]
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
    location = "*"

    # Check what channel codes and locations to use from only_record_ids if provided
    if only_record_ids is not None:
        # Check that we only have 1 record_id
        assert (
            len(only_record_ids) == 1
        ), "Multiple record_ids for the same event_sta combo"
        # Get the channel and location to use
        channel_codes = (
            only_record_ids["record_id"].str.split("_").str[-2].values[0] + "?"
        )
        location = only_record_ids["record_id"].str.split("_").str[-1].values[0]

    # Get the waveforms with multiple retries when IncompleteReadError occurs
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                st = client.get_waveforms(
                    net,
                    sta,
                    location,
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
        except FDSNServiceUnavailableException:
            print(f"Error getting waveforms for {net}.{sta}")
            print("Service temporarily unavailable")
            print("HTTP Status code: 503")
            print("Retrying in 2 minutes...")
            time.sleep(120)  # Wait for 2 minutes before retrying
        except Exception as e:  # noqa: BLE001
            print(f"Unexpected error getting waveforms for {net}.{sta}")
            print(e)
            return None
    return st


def split_stream_into_mseeds(st: Stream, unique_channels: Iterable, event_id: str):
    """
    Split the stream object into multiple mseed files based on the unique channel and location

    Parameters
    ----------
    st : Stream
        The stream object containing the waveform data for every channel and location
    unique_channels : Iterable
        An Iterable of tuples containing the unique channel and location for each mseed file created
        [(channel, location), ...]
    event_id: str
        The event id which is used if there is a raised issue with the mseed file

    Returns
    -------
    list
        A list of stream objects containing the waveform data for each mseed file created
    """
    mseeds = []
    raised_issues = []
    for chan, loc in unique_channels:
        # Each unique channel and location pair is a new mseed file
        st_new = st.select(location=loc, channel=f"{chan}?")
        record_id = f"{event_id}_{st_new[0].stats.station}_{st_new[0].stats.channel[:2]}_{st_new[0].stats.location}"

        if len(st_new) > 3:
            # Check if all the sample rates are the same
            samples = [tr.stats.sampling_rate for tr in st_new]
            if len(set(samples)) > 1:
                # If they are different take the highest and resample with the others using interpolation
                st_new = st_new.select(sampling_rate=max(samples))
                st_new.merge(fill_value="interpolate")
                raised_issues.append(
                    [record_id, "Split stream, different sample rates"]
                )

        # Check the final length of the traces
        if len(st_new) != 3:
            raised_issues.append([record_id, "Unknown issue, multiple traces"])
            continue

        # Ensure traces all have the same length
        starttime_trim = max([tr.stats.starttime for tr in st_new])
        endtime_trim = min([tr.stats.endtime for tr in st_new])
        # Check that the start time is before the end time
        if starttime_trim > endtime_trim:
            raised_issues.append(
                [record_id, "Unknown issue, start time after end time"]
            )
            continue
        st_new.trim(starttime_trim, endtime_trim)

        mseeds.append(st_new)

    return mseeds, raised_issues


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
        start_time = mseedlib.timestr2nstime(f"{trace.stats.starttime.isoformat()}Z")
        sourceid = f"FDSN:{trace.stats.network}_{trace.stats.station}_{trace.stats.location}_{'_'.join(trace.stats.channel)}"
        mstl.add_data(
            sourceid=sourceid,
            data_samples=trace.data,
            sample_type="i",
            sample_rate=trace.stats.sampling_rate,
            start_time=start_time,
        )

    with open(output_file, "wb") as f:
        mstl.pack(
            lambda record, handler_data: handler_data["fh"].write(record),
            {"fh": f},
            flush_data=True,
            format_version=2,
        )


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
