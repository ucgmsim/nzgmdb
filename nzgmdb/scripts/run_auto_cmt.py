"""
Run the automated multi-run 1-D CMT pipeline for a single event, and clean up CMT run folders.
"""

import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import requests
import typer
from auto_cmt.run_cmt import (
    CMT_SOLUTION_FILENAME,
    DEFAULT_NZ_3DVM_PATH,
    DEFAULT_THREADS,
    build_nz_velocity_grid,
    load_event_row,
    run_cmt_inversion,
)
from auto_cmt.timing import TIMINGS_FILENAME as STAGE_TIMINGS_FILENAME

from nzgmdb.management.cmt_cleanup import cleanup_cmt_run, find_cmt_run_dirs
from nzgmdb.management.timing import TIMINGS_FILENAME, StepTimer, safe_write_timings
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)

# Run folder names, in selection tie-break order
RUN_ALL_STATIONS = "all_stations"
RUN_AZIMUTH = "azimuth"
RUN_DROP_OUTLIERS_DEV = "drop_outliers_dev"

EVENT_CSV_FILENAME = "event.csv"
STATION_VR_FILENAME = "station_vr.csv"
CMT_SELECTION_FILENAME = "cmt_selection.csv"
RUN_LOG_FILENAME = "cmt_run.log"

# Outlier station defaults (see find_outlier_stations)
DEFAULT_OUTLIER_Z = 5.0
DEFAULT_OUTLIER_VR_FLOOR = -1.0
MIN_STATIONS_AFTER_DROP = 5

# Velocity grid shared with forked worker processes, set in run_cmt_pipeline
_NZ_GRID = None


def build_event_csv(event_id: str, output_ffp: Path) -> Path:
    """
    Build a one-row event CSV from the GeoNet quake API.

    Parameters
    ----------
    event_id : str
        GeoNet event/public ID.
    output_ffp : Path
        Path to write the event CSV to.

    Returns
    -------
    Path
        Path to the written event CSV.
    """
    r = requests.get(
        f"https://api.geonet.org.nz/quake/{event_id}",
        headers={"Accept": "application/vnd.geo+json;version=2"},
    )
    r.raise_for_status()
    features = r.json()["features"]
    if not features:
        raise ValueError(f"Event {event_id} not found in the GeoNet quake API")
    feature = features[0]

    pd.DataFrame(
        [
            {
                "evid": event_id,
                "datetime": feature["properties"]["time"],
                "lat": feature["geometry"]["coordinates"][1],
                "lon": feature["geometry"]["coordinates"][0],
                "depth": feature["properties"]["depth"],
                "mag": feature["properties"]["magnitude"],
            }
        ]
    ).to_csv(output_ffp, index=False)
    return output_ffp


def compute_station_vr(station_df: pd.DataFrame) -> pd.Series:
    """
    Compute the mean variance reduction per station over its used components.

    Parameters
    ----------
    station_df : pd.DataFrame
        BayesISOLA station_df.csv table, one row per station component.

    Returns
    -------
    pd.Series
        Station VR indexed by station ID (NET.STA.LOC, "--" for no location).
    """
    used_df = station_df[station_df["used"].astype(bool)].dropna(
        subset=["variance_reduction"]
    )
    location = used_df["location"].fillna("").astype(str).str.strip()
    station_id = (
        used_df["network"].astype(str).str.strip()
        + "."
        + used_df["station"].astype(str).str.strip()
        + "."
        + location.where(location != "", "--")
    )
    return used_df.groupby(station_id)["variance_reduction"].mean()


def read_station_df(run_dir: Path) -> pd.DataFrame:
    """
    Read a run's station_df.csv, keeping station codes as strings.

    Parameters
    ----------
    run_dir : Path
        CMT run folder.

    Returns
    -------
    pd.DataFrame
        Station/component fit table.
    """
    return pd.read_csv(
        run_dir / "results" / "station_df.csv",
        dtype={"network": str, "station": str, "location": str},
    )


def robust_z_score(station_vr: pd.Series) -> pd.Series:
    """
    Compute the robust (median / MAD) z-score of each station's VR.

    Parameters
    ----------
    station_vr : pd.Series
        Station VR indexed by station ID.

    Returns
    -------
    pd.Series
        Robust z-score indexed by station ID.
    """
    median = station_vr.median()
    # Floor the MAD so identical VRs don't divide by zero
    mad = max(1.4826 * (station_vr - median).abs().median(), 1e-6)
    return (station_vr - median) / mad


def find_outlier_stations(
    station_vr_all: pd.Series,
    station_vr_azimuth: pd.Series,
    outlier_z: float = DEFAULT_OUTLIER_Z,
    outlier_vr_floor: float = DEFAULT_OUTLIER_VR_FLOOR,
) -> tuple[list[str], pd.DataFrame, str]:
    """
    Find stations that are VR outliers in both the all-stations and azimuth runs.

    A station is an outlier in a run when its robust z-score is below
    -outlier_z and its VR is below outlier_vr_floor. Only stations that are
    outliers in both runs are dropped.

    Parameters
    ----------
    station_vr_all : pd.Series
        Station VR from the all-stations run.
    station_vr_azimuth : pd.Series
        Station VR from the azimuth-control run.
    outlier_z : float, optional
        Robust z-score magnitude a station must fall below, by default 5.0.
    outlier_vr_floor : float, optional
        Station VR a station must fall below, by default -1.0.

    Returns
    -------
    dropped : list[str]
        Station IDs (NET.STA.LOC) to drop.
    station_vr_df : pd.DataFrame
        Per-station VR, z-score and outlier flags for both runs.
    message : str
        Note on the outcome, empty if nothing notable happened.
    """
    station_vr_df = pd.DataFrame(
        index=station_vr_all.index.union(station_vr_azimuth.index)
    )
    station_vr_df.index.name = "station_id"
    for name, station_vr in [
        (RUN_ALL_STATIONS, station_vr_all),
        (RUN_AZIMUTH, station_vr_azimuth),
    ]:
        z = robust_z_score(station_vr)
        outlier = (z < -outlier_z) & (station_vr < outlier_vr_floor)
        station_vr_df[f"vr_{name}"] = station_vr
        station_vr_df[f"z_{name}"] = z
        station_vr_df[f"outlier_{name}"] = outlier.reindex(
            station_vr_df.index, fill_value=False
        )
    outlier_cols = [f"outlier_{RUN_ALL_STATIONS}", f"outlier_{RUN_AZIMUTH}"]

    station_vr_df["dropped"] = station_vr_df[outlier_cols].all(axis=1)
    dropped = station_vr_df.index[station_vr_df["dropped"]].tolist()

    # Keep enough stations for the final inversion
    message = ""
    if dropped and len(station_vr_all) - len(dropped) < MIN_STATIONS_AFTER_DROP:
        message = f"Not dropping {dropped}: fewer than {MIN_STATIONS_AFTER_DROP} stations would remain"
        station_vr_df["dropped"] = False
        dropped = []

    return dropped, station_vr_df.reset_index(), message


@contextmanager
def _redirect_output(log_ffp: Path) -> Iterator[None]:
    """
    Redirect this process's stdout/stderr file descriptors to a log file.

    Works at the file descriptor level, so output from compiled code
    (e.g. Axitra) is captured as well.

    Parameters
    ----------
    log_ffp : Path
        Log file to append to.

    Yields
    ------
    None
        Output is redirected while inside the context.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_fds = [os.dup(1), os.dup(2)]
    with open(log_ffp, "a") as log_file:
        os.dup2(log_file.fileno(), 1)
        os.dup2(log_file.fileno(), 2)
        # Keep Python prints in order with compiled-code output
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_fds[0], 1)
            os.dup2(saved_fds[1], 2)
            for fd in saved_fds:
                os.close(fd)


def _summarise_run(run_dir: Path) -> dict[str, Any]:
    """
    Read the key results of a completed run from its results/ folder.

    Parameters
    ----------
    run_dir : Path
        Completed CMT run folder.

    Returns
    -------
    dict
        n_stations, variance_reduction, Mw, centroid_depth_km and on_grid_edge.
    """
    summary = pd.read_csv(run_dir / "results" / "solution_summary.csv").iloc[0]
    centroid = pd.read_csv(run_dir / "results" / "centroid_location.csv").iloc[0]
    return {
        "n_stations": len(compute_station_vr(read_station_df(run_dir))),
        "variance_reduction": float(summary["variance_reduction"]),
        "Mw": float(summary["Mw"]),
        "centroid_depth_km": float(centroid["centroid_depth_km"]),
        "on_grid_edge": bool(centroid["on_grid_edge"]),
    }


def run_single_cmt(
    run_name: str,
    event_row: pd.Series,
    run_dir: Path,
    real_time: bool,
    threads: int,
    azimuth_control: bool = False,
    drop_stations: list[str] | None = None,
    deviatoric: bool = False,
    cleanup: bool = True,
    profile: bool = False,
) -> dict[str, Any]:
    """
    Run one CMT inversion, catching any failure, and clean up on success.

    Uses the velocity grid in the module level _NZ_GRID. Output is written
    to run_dir/cmt_run.log, stage timings to run_dir/results/timings.csv.

    Parameters
    ----------
    run_name : str
        Name of the run (also its folder name).
    event_row : pd.Series
        Event row from load_event_row.
    run_dir : Path
        Output folder for this run.
    real_time : bool
        Query GeoNet's near-real-time FDSN client.
    threads : int
        Axitra thread count.
    azimuth_control : bool, optional
        Apply azimuthal station thinning, by default False.
    drop_stations : list[str], optional
        Stations to exclude, by default None.
    deviatoric : bool, optional
        Invert for a deviatoric moment tensor only, by default False.
    cleanup : bool, optional
        Run cleanup_cmt_run after a successful run, by default True.
    profile : bool, optional
        Write a cProfile dump to run_dir/results/profile.prof, by default False.

    Returns
    -------
    dict
        Run record for cmt_selection.csv, including inversion_s and cleanup_s
        timings.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run": run_name,
        "folder": str(run_dir),
        "status": "ok",
        "message": "",
        "deviatoric": deviatoric,
        "azimuth_control": azimuth_control,
        "dropped_stations": ";".join(drop_stations or []),
        "inversion_s": np.nan,
        "cleanup_s": np.nan,
    }
    with _redirect_output(run_dir / RUN_LOG_FILENAME):
        start = time.perf_counter()
        try:
            run_cmt_inversion(
                event_row,
                run_dir,
                _NZ_GRID,
                real_time=real_time,
                threads=threads,
                azimuth_control=azimuth_control,
                drop_stations=drop_stations,
                deviatoric=deviatoric,
                profile=profile,
            )
            record["inversion_s"] = time.perf_counter() - start
            record.update(_summarise_run(run_dir))
            if cleanup:
                start = time.perf_counter()
                freed = cleanup_cmt_run(run_dir)
                record["cleanup_s"] = time.perf_counter() - start
                print(f"Cleanup freed {freed / 1e9:.2f} GB")
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            record["status"] = "failed"
            record["message"] = f"{type(exc).__name__}: {exc}"
            if np.isnan(record["inversion_s"]):
                record["inversion_s"] = time.perf_counter() - start
    print(f"[run_auto_cmt] {run_name}: {record['status']} {record['message']}")
    return record


def select_best_run(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Select the successful run with the highest variance reduction.

    Ties go to the earliest run in records.

    Parameters
    ----------
    records : list[dict]
        Run records from run_single_cmt, in tie-break order.

    Returns
    -------
    pd.DataFrame
        Selection table with a boolean "selected" column.
    """
    selection_df = pd.DataFrame(records)
    if "variance_reduction" not in selection_df:
        selection_df["variance_reduction"] = np.nan
    selection_df["selected"] = False
    ok = selection_df["status"] == "ok"
    if ok.any():
        best = selection_df.loc[ok, "variance_reduction"].idxmax()
        selection_df.loc[best, "selected"] = True
    return selection_df


def build_timings_df(
    output_dir: Path, timer: StepTimer, records: list[dict[str, Any]]
) -> pd.DataFrame:
    """
    Combine the pipeline step timings with each run's BayesISOLA stage timings.

    Parameters
    ----------
    output_dir : Path
        Event CMT output folder.
    timer : StepTimer
        Pipeline step timer.
    records : list[dict]
        Run records from run_single_cmt that finished, used for cleanup times.
        Runs without a record (e.g. a crashed worker) are still included if
        they wrote stage timings.

    Returns
    -------
    pd.DataFrame
        Columns run ("pipeline" or the run name), stage, calls, seconds
        (exclusive), inclusive_s and percent (of the pipeline total or of
        that run's total).
    """
    pipeline_df = timer.to_dataframe().rename(columns={"step": "stage"})
    pipeline_df.insert(0, "run", "pipeline")
    pipeline_df["calls"] = 1
    pipeline_df["inclusive_s"] = pipeline_df["seconds"]
    timing_dfs = [pipeline_df]

    cleanup_s = {record["run"]: record["cleanup_s"] for record in records}
    for run_name in [RUN_ALL_STATIONS, RUN_AZIMUTH, RUN_DROP_OUTLIERS_DEV]:
        stage_ffp = output_dir / run_name / "results" / STAGE_TIMINGS_FILENAME
        if not stage_ffp.is_file():
            continue
        stage_df = pd.read_csv(stage_ffp).rename(columns={"exclusive_s": "seconds"})
        stage_df.insert(0, "run", run_name)
        # Cleanup happens after the inversion, so it isn't part of the run's total
        if not np.isnan(cleanup_s.get(run_name, np.nan)):
            stage_df.loc[len(stage_df)] = {
                "run": run_name,
                "stage": "cleanup",
                "calls": 1,
                "seconds": round(cleanup_s[run_name], 2),
                "inclusive_s": round(cleanup_s[run_name], 2),
            }
        timing_dfs.append(stage_df)

    return pd.concat(timing_dfs, ignore_index=True)[
        ["run", "stage", "calls", "seconds", "inclusive_s", "percent"]
    ]


@cli.from_docstring(app)
def run_cmt_pipeline(
    event_id: Annotated[str, typer.Argument()],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    event_csv: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    real_time: Annotated[bool, typer.Option(is_flag=True)] = False,
    threads: Annotated[int, typer.Option()] = DEFAULT_THREADS,
    outlier_z: Annotated[float, typer.Option()] = DEFAULT_OUTLIER_Z,
    outlier_vr_floor: Annotated[float, typer.Option()] = DEFAULT_OUTLIER_VR_FLOOR,
    nz_3dvm_path: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = DEFAULT_NZ_3DVM_PATH,
    skip_cleanup: Annotated[bool, typer.Option(is_flag=True)] = False,
    profile: Annotated[bool, typer.Option(is_flag=True)] = False,
) -> pd.DataFrame:
    """
    Run the three-run CMT pipeline for one event and select the best solution.

    Runs all-stations and azimuth-control inversions in parallel, drops
    stations that are VR outliers in both, runs a deviatoric inversion
    without them, then selects the run with the highest variance reduction.

    Parameters
    ----------
    event_id : str
        GeoNet event/public ID.
    output_dir : Path
        Event CMT output folder (e.g. event_dir/cmt_1d).
    event_csv : Path, optional
        Event CSV (evid, datetime, lat, lon, depth, mag), e.g. the NZGMDB
        earthquake source table. If None, built from the GeoNet quake API.
    real_time : bool, optional
        Query GeoNet's near-real-time FDSN client, for just-happened events.
    threads : int, optional
        Axitra thread count for each run.
    outlier_z : float, optional
        Robust z-score magnitude a station must fall below to be an outlier.
    outlier_vr_floor : float, optional
        Station VR a station must fall below to be an outlier.
    nz_3dvm_path : Path, optional
        3-D NZ velocity model CSV.
    skip_cleanup : bool, optional
        Keep all run files (no cleanup), for debugging.
    profile : bool, optional
        Write a cProfile dump for each run to <run>/results/profile.prof.

    Returns
    -------
    pd.DataFrame
        Selection table, also written to cmt_selection.csv. Step and stage
        timings are written to timings.csv.
    """
    global _NZ_GRID

    timer = StepTimer()
    # Finished run records, so timings can be written whatever step fails
    records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        event_csv_ffp = output_dir / EVENT_CSV_FILENAME
        if event_csv is None:
            build_event_csv(event_id, event_csv_ffp)
        elif event_csv.resolve() != event_csv_ffp.resolve():
            shutil.copy(event_csv, event_csv_ffp)
        event_row = load_event_row(event_id, event_csv_ffp)
        timer.lap("event_csv")

        print("[run_auto_cmt] building velocity model grid")
        _NZ_GRID = build_nz_velocity_grid(nz_3dvm_path)
        timer.lap("velocity_grid")

        run_kwargs = {
            "event_row": event_row,
            "real_time": real_time,
            "threads": threads,
            "cleanup": not skip_cleanup,
            "profile": profile,
        }

        # Runs 1 and 2 in parallel, forked workers inherit _NZ_GRID
        print("[run_auto_cmt] running all_stations and azimuth runs")
        with ProcessPoolExecutor(
            max_workers=2, mp_context=mp.get_context("fork")
        ) as pool:
            all_future = pool.submit(
                run_single_cmt,
                RUN_ALL_STATIONS,
                run_dir=output_dir / RUN_ALL_STATIONS,
                **run_kwargs,
            )
            azimuth_future = pool.submit(
                run_single_cmt,
                RUN_AZIMUTH,
                run_dir=output_dir / RUN_AZIMUTH,
                azimuth_control=True,
                **run_kwargs,
            )
            all_record, azimuth_record = all_future.result(), azimuth_future.result()
        records += [all_record, azimuth_record]
        timer.lap("runs_all_stations_azimuth")

        # Outlier stations need both runs to agree
        dropped = []
        if all_record["status"] == "ok" and azimuth_record["status"] == "ok":
            dropped, station_vr_df, message = find_outlier_stations(
                compute_station_vr(read_station_df(output_dir / RUN_ALL_STATIONS)),
                compute_station_vr(read_station_df(output_dir / RUN_AZIMUTH)),
                outlier_z,
                outlier_vr_floor,
            )
            station_vr_df.to_csv(output_dir / STATION_VR_FILENAME, index=False)
        else:
            message = "Not dropping stations: all_stations or azimuth run failed"
        print(f"[run_auto_cmt] dropping stations: {dropped} {message}")
        timer.lap("outlier_detection")

        # Run 3 in a fresh worker so its output redirect stays isolated
        with ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context("fork")
        ) as pool:
            drop_record = pool.submit(
                run_single_cmt,
                RUN_DROP_OUTLIERS_DEV,
                run_dir=output_dir / RUN_DROP_OUTLIERS_DEV,
                drop_stations=dropped,
                deviatoric=True,
                **run_kwargs,
            ).result()
        records.append(drop_record)
        timer.lap("run_drop_outliers_dev")
        if message:
            drop_record["message"] = "; ".join(
                m for m in [message, drop_record["message"]] if m
            )

        selection_df = select_best_run(records)
        selection_df.to_csv(output_dir / CMT_SELECTION_FILENAME, index=False)
        print(selection_df.to_string())

        selected = selection_df["selected"].any()
        if selected:
            selected_dir = Path(
                selection_df.loc[selection_df["selected"], "folder"].iloc[0]
            )
            shutil.copy(
                selected_dir / CMT_SOLUTION_FILENAME, output_dir / CMT_SOLUTION_FILENAME
            )
            print(f"[run_auto_cmt] selected {selected_dir.name}")
        timer.lap("selection")

        if not selected:
            raise RuntimeError(f"All CMT runs failed for {event_id}, see {output_dir}")

        return selection_df
    finally:
        timer.lap("other")
        try:
            timings_df = build_timings_df(output_dir, timer, records)
        except (OSError, ValueError, KeyError) as exc:
            print(f"[run_auto_cmt] could not combine run timings: {exc}")
            timings_df = timer.to_dataframe()
        safe_write_timings(timings_df, output_dir / TIMINGS_FILENAME)
        print(timings_df.to_string(index=False))


@cli.from_docstring(app)
def cleanup_cmt(
    path: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    dry_run: Annotated[bool, typer.Option(is_flag=True)] = False,
) -> None:
    """
    Clean up every CMT run folder at or below a path.

    Parameters
    ----------
    path : Path
        A run folder, an event's CMT folder or a folder of events.
    dry_run : bool, optional
        Only print what would be deleted.
    """
    run_dirs = find_cmt_run_dirs(path)
    total = 0
    for run_dir in run_dirs:
        freed = cleanup_cmt_run(run_dir, dry_run=dry_run)
        total += freed
        print(f"{run_dir}: {freed / 1e9:.2f} GB")
    action = "would free" if dry_run else "freed"
    print(f"{len(run_dirs)} run folder(s), {action} {total / 1e9:.2f} GB")


if __name__ == "__main__":
    app()
