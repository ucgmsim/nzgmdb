"""Check how much of a time window has been downloaded for each temporary array station."""

from pathlib import Path

import pandas as pd
from obspy import UTCDateTime

from nzgmdb.tmp_array import file_layout
from nzgmdb.tmp_array.download import month_chunks


def check_row(
    row: pd.Series, data_dir: Path, start: UTCDateTime, end: UTCDateTime
) -> dict:
    """
    Summarise the downloaded files for one station row within the window.

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
        Status (complete, partial, missing, outside_window), chunk counts and
        the number of overlapping files (overlaps give duplicate traces in extraction)
    """
    window = file_layout.row_window(row, start, end)
    if window is None:
        return {"status": "outside_window"}

    mseed_dir = file_layout.record_dir(data_dir, file_layout.WAVEFORM_DIRNAME, row)
    chunks = month_chunks(*window)
    n_done = sum(file_layout.has_z_file_ending_at(mseed_dir, c[1]) for c in chunks)

    # Count files whose time range overlaps the previous file of the same channel
    parsed = (
        [p for f in mseed_dir.glob("*.mseed") if (p := file_layout.parse_mseed_name(f))]
        if mseed_dir.is_dir()
        else []
    )
    n_overlapping = 0
    for _, group in pd.DataFrame(parsed, columns=["cha", "start", "end"]).groupby(
        "cha"
    ):
        group = group.sort_values("start")
        n_overlapping += int(
            (
                group["start"].iloc[1:].to_numpy() < group["end"].iloc[:-1].to_numpy()
            ).sum()
        )

    if n_done == len(chunks):
        status = "complete"
    elif parsed:
        status = "partial"
    else:
        status = "missing"
    return {
        "status": status,
        "window_start": str(window[0]),
        "window_end": str(window[1]),
        "n_chunks": len(chunks),
        "n_chunks_done": n_done,
        "n_files": len(parsed),
        "n_overlapping_files": n_overlapping,
    }


def check_stations(
    station_df: pd.DataFrame, data_dir: Path, start: UTCDateTime, end: UTCDateTime
) -> pd.DataFrame:
    """
    Check every station row and print a summary.

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
        Station table with the check_row columns added
    """
    check_df = pd.DataFrame(
        [check_row(row, data_dir, start, end) for _, row in station_df.iterrows()],
        index=station_df.index,
    )
    result_df = pd.concat([station_df, check_df], axis=1)

    print(f"\nDownload status for {start} -> {end}")
    print(result_df["status"].value_counts().to_string())
    n_overlapping = int(
        result_df.get("n_overlapping_files", pd.Series()).fillna(0).sum()
    )
    if n_overlapping:
        print(
            f"WARNING: {n_overlapping} overlapping files found, see n_overlapping_files"
        )

    cols = ["provider", "net", "sta", "loc", "chan", "n_chunks_done", "n_chunks"]
    incomplete = result_df[result_df["status"].isin(["partial", "missing"])]
    if not incomplete.empty:
        print("\nIncomplete rows")
        print(incomplete[cols].to_string(index=False))
    return result_df
