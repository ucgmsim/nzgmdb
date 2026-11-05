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
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import matplotlib
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

from nzgmdb.management.background import CMT_FAILED_STEP_MARKER
from nzgmdb.management.cmt_cleanup import (
    GF_STORE_DIRNAME,
    cleanup_cmt_run,
    cleanup_gf_store,
    find_cmt_run_dirs,
    find_gf_store_dirs,
)
from nzgmdb.management.slack import CMT_EMOJI, EventThread
from nzgmdb.management.timing import (
    TIMINGS_FILENAME,
    StepTimer,
    format_duration,
    safe_write_timings,
)
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)

# Run folder names, in selection tie-break order
RUN_ALL_STATIONS = "all_stations"
RUN_AZIMUTH = "azimuth"
RUN_DROP_OUTLIERS_DEV = "drop_outliers_dev"
N_RUNS = 3

EVENT_CSV_FILENAME = "event.csv"
STATION_VR_FILENAME = "station_vr.csv"
CMT_SELECTION_FILENAME = "cmt_selection.csv"
RUN_LOG_FILENAME = "cmt_run.log"

# Seconds between checks for a run's cmt_solution.csv
SOLUTION_POLL_S = 0.5

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


def new_run_record(
    run_name: str,
    run_dir: Path,
    azimuth_control: bool = False,
    drop_stations: list[str] | None = None,
    deviatoric: bool = False,
) -> dict[str, Any]:
    """
    Create the cmt_selection.csv record of a run before it starts.

    Parameters
    ----------
    run_name : str
        Name of the run (also its folder name).
    run_dir : Path
        Output folder for this run.
    azimuth_control : bool, optional
        Apply azimuthal station thinning, by default False.
    drop_stations : list[str], optional
        Stations to exclude, by default None.
    deviatoric : bool, optional
        Invert for a deviatoric moment tensor only, by default False.

    Returns
    -------
    dict
        Run record, status "ok" until the run fails.
    """
    return {
        "run": run_name,
        "folder": str(run_dir),
        "status": "ok",
        "message": "",
        "deviatoric": deviatoric,
        "azimuth_control": azimuth_control,
        "dropped_stations": ";".join(drop_stations or []),
        "solution_s": np.nan,
        "run_s": np.nan,
        "cleanup_s": np.nan,
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
    station_df: Path | None = None,
    gf_store_dir: Path | None = None,
    profile: bool = False,
) -> dict[str, Any]:
    """
    Run one CMT inversion, including its figures and report, catching any failure.

    Runs in a forked worker and uses the velocity grid in the module level
    _NZ_GRID. Output is written to run_dir/cmt_run.log, stage timings to
    run_dir/results/timings.csv. cmt_solution.csv is written as soon as the
    solution exists, before the figures.

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
        Axitra/inversion processes.
    azimuth_control : bool, optional
        Apply azimuthal station thinning, by default False.
    drop_stations : list[str], optional
        Stations to exclude, by default None.
    deviatoric : bool, optional
        Invert for a deviatoric moment tensor only, by default False.
    station_df : Path, optional
        stations_downloaded.csv of an earlier run of the event, whose
        waveforms are reused instead of downloading again, by default None.
    gf_store_dir : Path, optional
        Green's function store shared by the runs of the event, by default None.
    profile : bool, optional
        Write a cProfile dump to run_dir/results/profile.prof, by default False.

    Returns
    -------
    dict
        status ("ok" or "failed"), message and run_s (seconds for the whole run).
    """
    result = {"status": "ok", "message": "", "run_s": np.nan}
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
                station_df=station_df,
                gf_store_dir=gf_store_dir,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            result["status"] = "failed"
            result["message"] = f"{type(exc).__name__}: {exc}"
        result["run_s"] = time.perf_counter() - start
    print(f"[run_auto_cmt] {run_name}: {result['status']} {result['message']}")
    return result


def apply_run_result(record: dict[str, Any], future: Future) -> None:
    """
    Update a run record with the result of its finished worker.

    A failure after cmt_solution.csv was written (e.g. in the figures) keeps
    the run's solution and is only noted in the message.

    Parameters
    ----------
    record : dict
        Run record from new_run_record, updated in place.
    future : Future
        Finished future of run_single_cmt.
    """
    try:
        result = future.result()
    except Exception as exc:  # noqa: BLE001
        # e.g. the worker was killed
        result = {
            "status": "failed",
            "message": f"{type(exc).__name__}: {exc}",
            "run_s": np.nan,
        }
    record["run_s"] = result["run_s"]
    if result["status"] == "ok":
        return
    if (Path(record["folder"]) / CMT_SOLUTION_FILENAME).is_file():
        record["message"] = "; ".join(
            m
            for m in [record["message"], f"figures/report failed: {result['message']}"]
            if m
        )
    else:
        record["status"] = "failed"
        record["message"] = result["message"]


def run_until_solution(
    pool: ProcessPoolExecutor, record: dict[str, Any], **run_kwargs: Any
) -> Future:
    """
    Start a run in a worker and wait until it has written its solution.

    The run's figures and report keep drawing in the background, so the next
    run can start straight away.

    Parameters
    ----------
    pool : ProcessPoolExecutor
        Forked worker pool, with a worker free for this run.
    record : dict
        Run record from new_run_record, updated in place with solution_s, the
        status and the solution summary.
    **run_kwargs : Any
        Passed to run_single_cmt.

    Returns
    -------
    Future
        Future of the run, done once its figures and report are finished.
    """
    run_dir = Path(record["folder"])
    run_dir.mkdir(parents=True, exist_ok=True)
    solution_ffp = run_dir / CMT_SOLUTION_FILENAME
    # A solution from an earlier attempt would look like this run finished
    solution_ffp.unlink(missing_ok=True)

    start = time.perf_counter()
    future = pool.submit(
        run_single_cmt,
        record["run"],
        run_dir=run_dir,
        azimuth_control=record["azimuth_control"],
        drop_stations=[s for s in record["dropped_stations"].split(";") if s],
        deviatoric=record["deviatoric"],
        **run_kwargs,
    )
    while not future.done() and not solution_ffp.is_file():
        time.sleep(SOLUTION_POLL_S)
    record["solution_s"] = time.perf_counter() - start

    if not solution_ffp.is_file():
        # Finished without a solution, so it failed
        apply_run_result(record, future)
    if record["status"] == "ok":
        try:
            record.update(_summarise_run(run_dir))
        except (OSError, KeyError, IndexError, ValueError) as exc:
            record["status"] = "failed"
            record["message"] = f"Could not read results: {type(exc).__name__}: {exc}"
    return future


def format_run_record(record: dict[str, Any]) -> str:
    """
    Format a run's solution for a Slack reply.

    Parameters
    ----------
    record : dict
        Run record after run_until_solution.

    Returns
    -------
    str
        Run name, time to its solution and its key results, or its error if
        it failed.
    """
    duration = format_duration(record["solution_s"])
    if record["status"] != "ok":
        return f"❌ CMT run `{record['run']}` failed after {duration}: `{record['message']}`"
    message = (
        f"✅ CMT run `{record['run']}` solution in {duration}: "
        f"VR {record['variance_reduction']:.2f}, Mw {record['Mw']:.2f}, "
        f"depth {record['centroid_depth_km']:.1f} km, {record['n_stations']} stations"
    )
    if record["on_grid_edge"]:
        message += " ⚠️ centroid on grid edge"
    return message


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
        Run records from new_run_record, used for cleanup times.
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


def cleanup_event_runs(
    output_dir: Path, records: list[dict[str, Any]], gf_store_dir: Path | None
) -> None:
    """
    Clean up the successful run folders and delete the shared Green's function store.

    Only called once every run has finished, as later runs read the
    all_stations waveforms and all runs share the store.

    Parameters
    ----------
    output_dir : Path
        Event CMT output folder.
    records : list[dict]
        Run records, cleanup_s is set in place for each cleaned run.
    gf_store_dir : Path, optional
        Shared Green's function store to delete, None if not used.
    """
    for record in records:
        if record["status"] != "ok":
            continue
        start = time.perf_counter()
        try:
            freed = cleanup_cmt_run(Path(record["folder"]))
            print(f"[run_auto_cmt] cleanup {record['run']} freed {freed / 1e9:.2f} GB")
        except (OSError, ValueError) as exc:
            print(f"[run_auto_cmt] could not clean up {record['run']}: {exc}")
        record["cleanup_s"] = time.perf_counter() - start
    if gf_store_dir is not None and gf_store_dir.is_dir():
        try:
            freed = cleanup_gf_store(gf_store_dir)
            print(f"[run_auto_cmt] deleted {gf_store_dir} ({freed / 1e9:.2f} GB)")
        except OSError as exc:
            print(f"[run_auto_cmt] could not delete {gf_store_dir}: {exc}")
    print(f"[run_auto_cmt] cleaned up {output_dir}")


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
    share_work: Annotated[bool, typer.Option()] = True,
    skip_cleanup: Annotated[bool, typer.Option(is_flag=True)] = False,
    profile: Annotated[bool, typer.Option(is_flag=True)] = False,
    slack_thread_ts: Annotated[str | None, typer.Option()] = None,
    slack_summary: Annotated[str, typer.Option()] = "",
) -> pd.DataFrame:
    """
    Run the three-run CMT pipeline for one event and select the best solution.

    Runs all-stations, then azimuth-control, then a deviatoric inversion
    without the stations that are VR outliers in both, one after another with
    all threads each (as auto_cmt.run_event). Each run's figures and report are
    drawn in the background while the next run starts. The run with the
    highest variance reduction is selected as soon as all solutions exist.

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
        Axitra/inversion processes for each run.
    outlier_z : float, optional
        Robust z-score magnitude a station must fall below to be an outlier.
    outlier_vr_floor : float, optional
        Station VR a station must fall below to be an outlier.
    nz_3dvm_path : Path, optional
        3-D NZ velocity model CSV.
    share_work : bool, optional
        Reuse the all_stations waveforms in later runs and share one Green's
        function store (gf_store/) between runs. Does not change any result.
    skip_cleanup : bool, optional
        Keep all run files and the Green's function store, for debugging.
    profile : bool, optional
        Write a cProfile dump for each run to <run>/results/profile.prof.
    slack_thread_ts : str, optional
        Slack thread of the event to reply into when each run starts and
        has its solution. If None, no Slack messages are sent.
    slack_summary : str, optional
        Event summary shown in the Slack thread's parent message.

    Returns
    -------
    pd.DataFrame
        Selection table, also written to cmt_selection.csv. Step and stage
        timings are written to timings.csv.
    """
    global _NZ_GRID

    # BayesISOLA plots with pyplot and forks inversion workers. An interactive
    # backend (TkAgg when a display is available) is not fork-safe and aborts
    # the run with "Tcl_AsyncDelete: async handler deleted by the wrong thread"
    matplotlib.use("Agg")

    timer = StepTimer()
    thread = EventThread(slack_thread_ts, slack_summary)
    # Run records and their futures, so timings and cleanup happen whatever step fails
    records = []
    futures = []
    output_dir.mkdir(parents=True, exist_ok=True)
    gf_store_dir = output_dir / GF_STORE_DIRNAME if share_work else None
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
            "gf_store_dir": gf_store_dir,
            "profile": profile,
        }

        # One worker per run, as each keeps drawing its figures after its
        # solution. Forked workers inherit _NZ_GRID
        with ProcessPoolExecutor(
            max_workers=N_RUNS, mp_context=mp.get_context("fork")
        ) as pool:
            # Run 1: all stations, downloads the waveforms
            all_record = new_run_record(RUN_ALL_STATIONS, output_dir / RUN_ALL_STATIONS)
            records.append(all_record)
            thread.reply(
                f"{CMT_EMOJI} CMT run `{RUN_ALL_STATIONS}` started (1/{N_RUNS}, "
                f"{threads} threads)"
            )
            futures.append(run_until_solution(pool, all_record, **run_kwargs))
            timer.lap(f"run_{RUN_ALL_STATIONS}")
            thread.reply(format_run_record(all_record))

            # Later runs reuse the all_stations waveforms when they exist
            downloaded_ffp = (
                output_dir / RUN_ALL_STATIONS / "metadata" / "stations_downloaded.csv"
            )
            station_df = (
                downloaded_ffp if share_work and downloaded_ffp.is_file() else None
            )
            reuse_note = (
                f", reusing `{RUN_ALL_STATIONS}` waveforms" if station_df else ""
            )

            # Run 2: azimuth control
            azimuth_record = new_run_record(
                RUN_AZIMUTH, output_dir / RUN_AZIMUTH, azimuth_control=True
            )
            records.append(azimuth_record)
            thread.reply(
                f"{CMT_EMOJI} CMT run `{RUN_AZIMUTH}` started (2/{N_RUNS}, "
                f"azimuthal thinning{reuse_note})"
            )
            futures.append(
                run_until_solution(
                    pool, azimuth_record, station_df=station_df, **run_kwargs
                )
            )
            timer.lap(f"run_{RUN_AZIMUTH}")
            thread.reply(format_run_record(azimuth_record))

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

            # Run 3: deviatoric without the outlier stations
            drop_record = new_run_record(
                RUN_DROP_OUTLIERS_DEV,
                output_dir / RUN_DROP_OUTLIERS_DEV,
                drop_stations=dropped,
                deviatoric=True,
            )
            records.append(drop_record)
            thread.reply(
                f"{CMT_EMOJI} CMT run `{RUN_DROP_OUTLIERS_DEV}` started (3/{N_RUNS}, "
                f"deviatoric{reuse_note}), dropping stations: {', '.join(dropped) or 'none'}"
                + (f" ({message})" if message else "")
            )
            futures.append(
                run_until_solution(
                    pool, drop_record, station_df=station_df, **run_kwargs
                )
            )
            timer.lap(f"run_{RUN_DROP_OUTLIERS_DEV}")
            thread.reply(format_run_record(drop_record))
            if message:
                drop_record["message"] = "; ".join(
                    m for m in [message, drop_record["message"]] if m
                )

            # Select as soon as all solutions exist, figures are still drawing
            selection_df = select_best_run(records)
            selected = selection_df["selected"].any()
            if selected:
                selected_row = selection_df[selection_df["selected"]].iloc[0]
                shutil.copy(
                    Path(selected_row["folder"]) / CMT_SOLUTION_FILENAME,
                    output_dir / CMT_SOLUTION_FILENAME,
                )
                print(f"[run_auto_cmt] selected {selected_row['run']}")
            timer.lap("selection")

            if not selected:
                raise RuntimeError(
                    f"All CMT runs failed for {event_id}, see {output_dir}"
                )

            solutions_s = timer.to_dataframe()["seconds"].iloc[-1]
            thread.reply(
                f"📌 All CMT solutions in {format_duration(solutions_s)}, "
                f"selected `{selected_row['run']}` "
                f"(VR {selected_row['variance_reduction']:.2f}). "
                "Finishing figures and reports"
            )
        # Leaving the pool waited for the background figures
        timer.lap("figures")

    except Exception:
        # Read by management.background for the Slack failure message
        last_step = timer.rows[-1]["step"] if timer.rows else "start"
        print(f"{CMT_FAILED_STEP_MARKER}{last_step}")
        raise
    finally:
        # All workers have finished once the pool is closed
        for record, future in zip(records, futures):
            apply_run_result(record, future)
        if not skip_cleanup:
            cleanup_event_runs(output_dir, records, gf_store_dir)
            timer.lap("cleanup")
        if records:
            select_best_run(records).to_csv(
                output_dir / CMT_SELECTION_FILENAME, index=False
            )
        timer.lap("other")
        try:
            timings_df = build_timings_df(output_dir, timer, records)
        except (OSError, ValueError, KeyError) as exc:
            print(f"[run_auto_cmt] could not combine run timings: {exc}")
            timings_df = timer.to_dataframe()
        # Only a summary line, so the log tail keeps any traceback
        safe_write_timings(timings_df, output_dir / TIMINGS_FILENAME)
        print(
            f"[run_auto_cmt] timings written to {output_dir / TIMINGS_FILENAME} "
            f"(total {format_duration(timer.to_dataframe()['seconds'].iloc[-1])})"
        )

    return select_best_run(records)


@cli.from_docstring(app)
def cleanup_cmt(
    path: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    dry_run: Annotated[bool, typer.Option(is_flag=True)] = False,
) -> None:
    """
    Clean up every CMT run folder and Green's function store at or below a path.

    Parameters
    ----------
    path : Path
        A run folder, an event's CMT folder or a folder of events.
    dry_run : bool, optional
        Only print what would be deleted.
    """
    run_dirs = find_cmt_run_dirs(path)
    gf_store_dirs = find_gf_store_dirs(path)
    total = 0
    for run_dir in run_dirs:
        freed = cleanup_cmt_run(run_dir, dry_run=dry_run)
        total += freed
        print(f"{run_dir}: {freed / 1e9:.2f} GB")
    for gf_store_dir in gf_store_dirs:
        freed = cleanup_gf_store(gf_store_dir, dry_run=dry_run)
        total += freed
        print(f"{gf_store_dir}: {freed / 1e9:.2f} GB")
    action = "would free" if dry_run else "freed"
    print(
        f"{len(run_dirs)} run folder(s), {len(gf_store_dirs)} Green's function "
        f"store(s), {action} {total / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    app()
