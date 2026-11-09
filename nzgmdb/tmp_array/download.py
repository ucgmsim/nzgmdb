"""
Bulk download continuous waveforms and StationXML for temporary array stations.

Data is requested in calendar month chunks so that runs over different time windows
line up and never write overlapping files, which lets new windows be added to an
existing data directory later.
"""

from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import (
    GlobalDomain,
    MassDownloader,
    Restrictions,
)

from nzgmdb.tmp_array import file_layout

# Smallest chunk to back off to when a provider rejects a request as too large
MIN_CHUNK_SECONDS = 3600
MAX_BACKOFF_ATTEMPTS = 4


def month_chunks(
    start: UTCDateTime, end: UTCDateTime
) -> list[tuple[UTCDateTime, UTCDateTime]]:
    """
    Split a time window on calendar month boundaries.

    Parameters
    ----------
    start : UTCDateTime
        Window start
    end : UTCDateTime
        Window end

    Returns
    -------
    list[tuple[UTCDateTime, UTCDateTime]]
        (chunk_start, chunk_end) pairs covering the window
    """
    chunks = []
    chunk_start = start
    while chunk_start < end:
        next_month = UTCDateTime(
            chunk_start.year + chunk_start.month // 12, chunk_start.month % 12 + 1, 1
        )
        chunk_end = min(next_month, end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunks


def is_too_large_error(error: Exception) -> bool:
    """
    Check if a download error is the provider rejecting the request size.

    Parameters
    ----------
    error : Exception
        Error raised by MassDownloader

    Returns
    -------
    bool
        True for HTTP 413 / manifest too large errors
    """
    text = f"{error!r} {error}"
    return any(
        msg in text
        for msg in ("Estimated manifest size", "Request Entity Too Large", "413")
    )


def download_chunk(
    row: pd.Series,
    chunk_start: UTCDateTime,
    chunk_end: UTCDateTime,
    mseed_dir: Path,
    xml_dir: Path,
):
    """
    Download one month chunk for a station row, halving the chunk length if rejected as too large.

    Parameters
    ----------
    row : pd.Series
        Station table row
    chunk_start : UTCDateTime
        Chunk start
    chunk_end : UTCDateTime
        Chunk end
    mseed_dir : Path
        Output folder for mseed files
    xml_dir : Path
        Output folder for StationXML files

    Raises
    ------
    Exception
        The MassDownloader error if it is not a size error or the chunk cannot shrink further
    """
    loc = "*" if row["loc"] == "NA" else row["loc"]
    chunk_length = int(chunk_end - chunk_start)

    for _ in range(MAX_BACKOFF_ATTEMPTS):
        restrictions = Restrictions(
            starttime=chunk_start,
            endtime=chunk_end,
            chunklength_in_sec=chunk_length,
            network=row["net"],
            station=row["sta"],
            location=loc,
            channel=f"{row['chan']}?",
            reject_channels_with_gaps=False,
            minimum_length=0.0,
            minimum_interstation_distance_in_m=0.0,
        )
        try:
            MassDownloader(providers=[row["provider"]]).download(
                GlobalDomain(),
                restrictions,
                mseed_storage=str(mseed_dir),
                stationxml_storage=str(xml_dir),
            )
            return
        except Exception as e:  # noqa: BLE001
            if not is_too_large_error(e) or chunk_length <= MIN_CHUNK_SECONDS:
                raise
            chunk_length = max(MIN_CHUNK_SECONDS, chunk_length // 2)
            print(f"  Request too large, retrying with {chunk_length}s chunks")
    raise RuntimeError(f"Request still too large after {MAX_BACKOFF_ATTEMPTS} attempts")


def download_row(
    row: pd.Series, data_dir: Path, start: UTCDateTime, end: UTCDateTime
) -> dict:
    """
    Download all missing month chunks for one station row within the window.

    Parameters
    ----------
    row : pd.Series
        Station table row
    data_dir : Path
        Root temporary array data directory
    start : UTCDateTime
        Window start
    end : UTCDateTime
        Window end

    Returns
    -------
    dict
        Result with status (ok, error, outside_window), chunk counts and any error
    """
    result = {
        "provider": row["provider"],
        "net": row["net"],
        "sta": row["sta"],
        "chan": row["chan"],
        "loc": row["loc"],
    }
    window = file_layout.row_window(row, start, end)
    if window is None:
        return result | {"status": "outside_window"}

    mseed_dir = file_layout.record_dir(data_dir, file_layout.WAVEFORM_DIRNAME, row)
    xml_dir = file_layout.record_dir(data_dir, file_layout.STATIONXML_DIRNAME, row)
    mseed_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    chunks = month_chunks(*window)
    todo = [c for c in chunks if not file_layout.has_z_file_ending_at(mseed_dir, c[1])]
    result |= {
        "window_start": str(window[0]),
        "window_end": str(window[1]),
        "n_chunks": len(chunks),
        "n_chunks_requested": len(todo),
    }

    errors = []
    for chunk_start, chunk_end in todo:
        print(f"  {chunk_start.date} -> {chunk_end.date}")
        try:
            download_chunk(row, chunk_start, chunk_end, mseed_dir, xml_dir)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{chunk_start.date}: {e}")

    # Months with no file after downloading had no data at the provider (or are restricted)
    n_without_data = sum(
        not file_layout.has_z_file_ending_at(mseed_dir, c[1]) for c in chunks
    )
    result["n_chunks_without_data"] = n_without_data
    if errors:
        return result | {"status": "error", "error": " | ".join(errors)}
    if n_without_data == len(chunks):
        return result | {"status": "no_data"}
    return result | {"status": "ok" if n_without_data == 0 else "partial"}


def download_stations(
    station_df: pd.DataFrame, data_dir: Path, start: UTCDateTime, end: UTCDateTime
) -> pd.DataFrame:
    """
    Download every station row for the window, skipping months that are already on disk.

    Parameters
    ----------
    station_df : pd.DataFrame
        Station table from get-stations
    data_dir : Path
        Root temporary array data directory
    start : UTCDateTime
        Window start
    end : UTCDateTime
        Window end

    Returns
    -------
    pd.DataFrame
        One result row per station row
    """
    results = []
    for idx, row in station_df.iterrows():
        print(
            f"[{idx + 1}/{len(station_df)}] {row['provider']} "
            f"{row['net']}.{row['sta']}.{row['loc']}.{row['chan']}?"
        )
        results.append(download_row(row, data_dir, start, end))
    return pd.DataFrame(results)
