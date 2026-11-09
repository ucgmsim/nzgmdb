"""
Track station data that could not be downloaded yet (embargoed or not yet released) and re-check it.

Every download run updates <data_dir>/pending_data.csv with rows that still have months
without data. `recheck-pending` retries only those rows and lists what became available,
so it can be run through the pipeline and merged into the database.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime

from nzgmdb.tmp_array import download

PENDING_NAME = "pending_data.csv"
KEY_COLUMNS = ["provider", "net", "sta", "chan", "loc", "window_start", "window_end"]
PENDING_COLUMNS = KEY_COLUMNS + [
    "restricted",
    "n_chunks",
    "n_chunks_without_data",
    "missing_reason",
    "first_seen",
    "last_checked",
]
PENDING_STATUSES = ["partial", "no_data", "embargoed"]


def read_pending(data_dir: Path) -> pd.DataFrame:
    """
    Read the pending data list, empty if it does not exist yet.

    Parameters
    ----------
    data_dir : Path
        Root temporary array data directory

    Returns
    -------
    pd.DataFrame
        Pending rows with PENDING_COLUMNS
    """
    pending_ffp = data_dir / PENDING_NAME
    if not pending_ffp.exists():
        return pd.DataFrame(columns=PENDING_COLUMNS)
    return pd.read_csv(pending_ffp, dtype=str, keep_default_na=False)


def update_pending(
    data_dir: Path, station_df: pd.DataFrame, results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Update the pending list from download results and write it.

    Rows that are now complete are removed, rows still missing months are added or
    updated, and rows that failed with an error are left as they were.

    Parameters
    ----------
    data_dir : Path
        Root temporary array data directory
    station_df : pd.DataFrame
        Station rows that were downloaded (for the restricted flag)
    results_df : pd.DataFrame
        Results from download.download_stations, same order as station_df

    Returns
    -------
    pd.DataFrame
        The updated pending list
    """
    pending_df = read_pending(data_dir)
    checked = results_df[~results_df["status"].isin(["error", "outside_window"])].copy()
    if checked.empty:
        return pending_df
    checked["restricted"] = station_df["restricted"].to_numpy()[checked.index]
    checked = checked.astype({"n_chunks": int, "n_chunks_without_data": int}).astype(
        str
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    checked["last_checked"] = now
    first_seen = pending_df.set_index(KEY_COLUMNS)["first_seen"]
    checked["first_seen"] = [
        first_seen.get(key, now)
        for key in checked[KEY_COLUMNS].itertuples(index=False, name=None)
    ]

    # Replace every checked key, then keep only those still missing data
    checked_keys = pd.MultiIndex.from_frame(checked[KEY_COLUMNS])
    pending_df = pending_df[
        ~pd.MultiIndex.from_frame(pending_df[KEY_COLUMNS]).isin(checked_keys)
    ]
    still_pending = checked[checked["status"].isin(PENDING_STATUSES)]
    pending_df = pd.concat([pending_df, still_pending[PENDING_COLUMNS]])
    pending_df = pending_df.sort_values(KEY_COLUMNS).reset_index(drop=True)
    pending_df.to_csv(data_dir / PENDING_NAME, index=False)
    return pending_df


def recheck_pending(
    data_dir: Path, providers: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retry every pending row and report which ones gained data.

    Parameters
    ----------
    data_dir : Path
        Root temporary array data directory
    providers : list[str], optional
        Only recheck these providers, by default all

    Returns
    -------
    pending_df : pd.DataFrame
        The updated pending list
    newly_available_df : pd.DataFrame
        Rows that gained data, with the number of months gained
    """
    pending_df = read_pending(data_dir)
    if providers:
        pending_df = pending_df[pending_df["provider"].isin(providers)]
    if pending_df.empty:
        print("Nothing pending")
        return read_pending(data_dir), pd.DataFrame()
    pending_df = pending_df.reset_index(drop=True)

    # Each pending row becomes a station row covering just its window
    station_df = pending_df.assign(
        start_date=pending_df["window_start"], end_date=pending_df["window_end"]
    )
    results = []
    for idx, row in station_df.iterrows():
        print(
            f"[{idx + 1}/{len(station_df)}] {row['provider']} "
            f"{row['net']}.{row['sta']}.{row['loc']}.{row['chan']}? "
            f"{row['window_start'][:10]} -> {row['window_end'][:10]}"
        )
        start, end = UTCDateTime(row["window_start"]), UTCDateTime(row["window_end"])
        results.append(download.download_row(row, data_dir, start, end))
    results_df = pd.DataFrame(results)

    gained = pending_df["n_chunks_without_data"].astype(int) - results_df[
        "n_chunks_without_data"
    ].fillna(pending_df["n_chunks_without_data"]).astype(int)
    newly_available_df = pending_df[KEY_COLUMNS].assign(
        n_chunks_gained=gained, status=results_df["status"]
    )
    newly_available_df = newly_available_df[gained > 0].reset_index(drop=True)

    return update_pending(data_dir, station_df, results_df), newly_available_df


def summarise_for_pipeline(newly_available_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group newly available rows into the sites and date range to re-run the pipeline for.

    Parameters
    ----------
    newly_available_df : pd.DataFrame
        Output of recheck_pending

    Returns
    -------
    pd.DataFrame
        One row per network with comma separated sites and the overall start and end
    """
    return (
        newly_available_df.groupby("net")
        .agg(
            sites=("sta", lambda s: ",".join(sorted(set(s)))),
            start=("window_start", "min"),
            end=("window_end", "max"),
        )
        .reset_index()
        .assign(
            start=lambda df: df["start"].map(lambda t: str(t)[:10]),
            end=lambda df: df["end"].map(lambda t: str(t)[:10]),
        )
    )


def describe_pending(pending_df: pd.DataFrame) -> str:
    """
    Make a short text summary of the pending list per network and reason.

    Parameters
    ----------
    pending_df : pd.DataFrame
        Pending list

    Returns
    -------
    str
        Summary table as text
    """
    if pending_df.empty:
        return "Nothing pending"
    return (
        pending_df.groupby(["provider", "net", "missing_reason"])
        .agg(
            n_rows=("sta", "size"),
            n_stations=("sta", "nunique"),
            months_missing=("n_chunks_without_data", lambda s: s.astype(int).sum()),
        )
        .reset_index()
        .to_string(index=False)
    )
