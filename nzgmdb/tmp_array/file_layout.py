"""
Shared folder layout, filename and request table helpers for temporary array data.

Layout under a data directory:
    waveforms/<net>/<net>_<sta>_<chan>_<loc>/<NET>.<STA>.<LOC>.<CHA>__<START>__<END>.mseed
    stationxml/<net>/<net>_<sta>_<chan>_<loc>/<NET>.<STA>.xml

The pipeline reads the waveforms folder with `extract-waveforms --tmp-array-dir`.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime

WAVEFORM_DIRNAME = "waveforms"
STATIONXML_DIRNAME = "stationxml"
STATION_COLUMNS = [
    "provider",
    "net",
    "sta",
    "lat",
    "lon",
    "elev",
    "chan",
    "loc",
    "start_date",
    "end_date",
]
MSEED_TIME_FORMAT = "%Y%m%dT%H%M%SZ"


def to_utc(value: datetime | str | UTCDateTime) -> UTCDateTime:
    """
    Convert a datetime or string to UTCDateTime (naive datetimes are treated as UTC).

    Parameters
    ----------
    value : datetime | str | UTCDateTime
        Time value to convert

    Returns
    -------
    UTCDateTime
        Converted time
    """
    return value if isinstance(value, UTCDateTime) else UTCDateTime(value)


def format_mseed_time(time: UTCDateTime) -> str:
    """
    Format a time the same way ObsPy's MassDownloader names its mseed files.

    Parameters
    ----------
    time : UTCDateTime
        Time to format

    Returns
    -------
    str
        Time formatted as YYYYMMDDTHHMMSSZ
    """
    return time.strftime(MSEED_TIME_FORMAT)


def read_station_csv(station_ffp: Path) -> pd.DataFrame:
    """
    Read a station request table written by `get-stations`.

    Parameters
    ----------
    station_ffp : Path
        Path to the station CSV

    Returns
    -------
    pd.DataFrame
        Station table with all columns as strings

    Raises
    ------
    ValueError
        If required columns are missing
    """
    station_df = pd.read_csv(station_ffp, dtype=str, keep_default_na=False)
    missing = sorted(set(STATION_COLUMNS) - set(station_df.columns))
    if missing:
        raise ValueError(f"{station_ffp} is missing columns: {missing}")
    return station_df


def row_window(
    row: pd.Series, start: UTCDateTime, end: UTCDateTime
) -> tuple[UTCDateTime, UTCDateTime] | None:
    """
    Clip a station row's operating dates to the requested window.

    Parameters
    ----------
    row : pd.Series
        Station table row
    start : UTCDateTime
        Window start
    end : UTCDateTime
        Window end
        Also used when the row has no end date (station still running)

    Returns
    -------
    tuple[UTCDateTime, UTCDateTime] | None
        Clipped (start, end), or None if the row does not overlap the window
    """
    row_start = UTCDateTime(row["start_date"]) if row["start_date"] else start
    row_end = UTCDateTime(row["end_date"]) if row["end_date"] else end
    clipped_start, clipped_end = max(row_start, start), min(row_end, end)
    if clipped_start >= clipped_end:
        return None
    return clipped_start, clipped_end


def record_dir(data_dir: Path, subdir: str, row: pd.Series) -> Path:
    """
    Get the folder for a station row's waveform or StationXML files.

    Parameters
    ----------
    data_dir : Path
        Root temporary array data directory
    subdir : str
        WAVEFORM_DIRNAME or STATIONXML_DIRNAME
    row : pd.Series
        Station table row

    Returns
    -------
    Path
        Path like <data_dir>/<subdir>/<net>/<net>_<sta>_<chan>_<loc>
    """
    net, sta, chan, loc = (
        row["net"].strip(),
        row["sta"].strip(),
        row["chan"].strip(),
        row["loc"].strip(),
    )
    return data_dir / subdir / net / f"{net}_{sta}_{chan}_{loc}"


def parse_mseed_name(mseed_ffp: Path) -> tuple[str, UTCDateTime, UTCDateTime] | None:
    """
    Parse channel, start and end from a MassDownloader mseed filename.

    Parameters
    ----------
    mseed_ffp : Path
        Path like NET.STA.LOC.CHA__START__END.mseed

    Returns
    -------
    tuple[str, UTCDateTime, UTCDateTime] | None
        (channel, start, end), or None if the name does not match
    """
    parts = mseed_ffp.stem.split("__")
    codes = parts[0].split(".")
    if mseed_ffp.suffix != ".mseed" or len(parts) != 3 or len(codes) != 4:
        return None
    try:
        return codes[3], UTCDateTime(parts[1]), UTCDateTime(parts[2])
    except ValueError:
        return None


def has_z_file_ending_at(mseed_dir: Path, end: UTCDateTime) -> bool:
    """
    Check if a vertical channel file ends exactly at the given time.

    This marks a chunk (or a full row) as downloaded.

    Parameters
    ----------
    mseed_dir : Path
        Folder of mseed files for one station row
    end : UTCDateTime
        Expected end time

    Returns
    -------
    bool
        True if a Z channel file ends at `end`
    """
    if not mseed_dir.is_dir():
        return False
    suffix = f"__{format_mseed_time(end)}.mseed"
    return any(
        f.name.endswith(suffix)
        and (parsed := parse_mseed_name(f))
        and parsed[0][-1] == "Z"
        for f in mseed_dir.iterdir()
    )
