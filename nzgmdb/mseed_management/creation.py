"""
This module contains functions for creating mseed files from the waveform data from the FDSN client
"""

from pathlib import Path

import mseedlib
import numpy as np
from obspy import Stream


def _coerce_mseed_samples(data: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Coerce trace samples to a MiniSEED-compatible dtype and return (samples, sample_type).

    Uses:
      - int32 -> "i"
      - float32 -> "f"
    """
    arr = np.asarray(data)

    # Ensure 1-D numeric
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if arr.dtype.kind in {"i", "u"}:
        arr_i32 = np.ascontiguousarray(arr.astype(np.int32, copy=False))
        return arr_i32, "i"

    # Default to float32 for floats/others numeric-like
    arr_f32 = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    return arr_f32, "f"


def write_stream_to_mseed(stream: Stream, output_file: Path) -> None:
    """
    Write an ObsPy Stream object to a MiniSEED file using mseedlib.
    """
    mstl = mseedlib.MSTraceList()

    for trace in stream:
        start_time = mseedlib.timestr2nstime(f"{trace.stats.starttime.isoformat()}Z")

        # Prefer channel as a string; avoid join() on a string (it inserts underscores between characters).
        channel = str(trace.stats.channel)
        location = str(trace.stats.location or "00")

        sourceid = (
            f"FDSN:{trace.stats.network}_{trace.stats.station}_{location}_{channel}"
        )

        samples, sample_type = _coerce_mseed_samples(trace.data)

        mstl.add_data(
            sourceid=sourceid,
            data_samples=samples,
            sample_type=sample_type,
            sample_rate=float(trace.stats.sampling_rate),
            start_time=start_time,
        )

    output_file.parent.mkdir(exist_ok=True, parents=True)
    with output_file.open("wb") as f:
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

    # If the location is empty, set it to "00" as the default value
    # based on the FDSN Source Indentifiers documentation (https://docs.fdsn.org/projects/source-identifiers/en/latest/location-codes.html)
    if location is None or location == "":
        location = "00"

    # Create the filename and add it to the output directory
    filename = f"{event_id}_{station}_{channel}_{location}.mseed"
    mseed_ffp = output_directory / filename
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write the mseed file
    write_stream_to_mseed(mseed, mseed_ffp)
