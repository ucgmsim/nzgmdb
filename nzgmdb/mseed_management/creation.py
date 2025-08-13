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

from nzgmdb.management import config as cfg
from oq_wrapper import constants, estimations, wrapper


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
