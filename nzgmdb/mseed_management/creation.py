"""
This module contains functions for creating mseed files from the waveform data from the FDSN client
"""

from pathlib import Path

import numpy as np
from obspy import Stream, Trace

from nzgmdb.management import file_structure


def write_stream_to_mseed(stream: Stream | Trace, output_file: Path):
    """
    Write an ObsPy Stream or Trace to a MiniSEED v2 file (STEIM1, 4096 byte records).

    Uses ObsPy's writer, as mseedlib's MSTraceList.add_data leaves a dangling
    pointer in the C trace segment that corrupts the heap of the process.

    Parameters
    ----------
    stream : Stream or Trace
        The Stream or Trace object to write to MiniSEED
    output_file : Path
        The path to the output MiniSEED file

    Raises
    ------
    ValueError
        If the trace data is not integer counts
    """
    stream = Stream([stream]) if isinstance(stream, Trace) else stream.copy()
    for trace in stream:
        if not np.issubdtype(trace.data.dtype, np.integer):
            raise ValueError(
                f"Unsupported sample type {trace.data.dtype} for {trace.id}, expected integer counts"
            )
        trace.data = trace.data.astype(np.int32, copy=False)

    stream.write(str(output_file), format="MSEED", encoding="STEIM1", reclen=4096)


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
    # Create the filename from the first trace and add it to the output directory
    stats = mseed[0].stats
    record_id = file_structure.get_record_id(
        event_id, station, stats.channel, stats.location
    )
    filename = f"{record_id}.mseed"
    mseed_ffp = output_directory / filename
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write the mseed file
    write_stream_to_mseed(mseed, mseed_ffp)
