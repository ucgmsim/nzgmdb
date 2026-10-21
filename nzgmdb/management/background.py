"""
Launch the multi-run 1-D CMT pipeline as a detached background subprocess, and
notify Slack when it finishes.
"""

import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

from nzgmdb.management.slack import reply_to_message_on_slack
from nzgmdb.management.timing import TIMINGS_FILENAME, format_duration

CMT_MODULE = "nzgmdb.scripts.run_auto_cmt"
CMT_COMMAND = "run-cmt-pipeline"
LOG_TAIL_LINES = 20

# Written by nzgmdb.scripts.run_auto_cmt into its output_dir. Kept as plain
# strings so importing this module doesn't import BayesISOLA.
CMT_SOLUTION_FILENAME = "cmt_solution.csv"
CMT_SELECTION_FILENAME = "cmt_selection.csv"

# Number of slowest BayesISOLA stages to report in Slack
TIMING_TOP_STAGES = 3


def _tail(path: Path, n_lines: int = LOG_TAIL_LINES) -> str:
    """
    Return the last n_lines of a text file.

    Parameters
    ----------
    path : Path
        File to read.
    n_lines : int, optional
        Number of trailing lines to return.

    Returns
    -------
    str
        The trailing lines joined with newlines, or a placeholder message
        if the file can't be read.
    """
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "(log file unavailable)"
    return "\n".join(lines[-n_lines:])


def _read_cmt_result_summary(output_dir: Path) -> str:
    """
    Build a short summary of a completed CMT pipeline for Slack.

    Reads the selected run and each run's variance reduction from
    cmt_selection.csv, and the moment magnitude, latitude, longitude and
    centroid depth from the selected cmt_solution.csv.

    Parameters
    ----------
    output_dir : Path
        Directory the CMT pipeline wrote its results to.

    Returns
    -------
    str
        Summary text, or an empty string if the files are missing or can't
        be parsed.
    """
    try:
        selection_df = pd.read_csv(output_dir / CMT_SELECTION_FILENAME)
        solution = pd.read_csv(output_dir / CMT_SOLUTION_FILENAME).iloc[0]
        selected = selection_df[selection_df["selected"]].iloc[0]
    except (OSError, KeyError, IndexError, ValueError) as exc:
        print(
            f"[management.background] could not read CMT results in {output_dir}: {exc}"
        )
        return ""

    others = ", ".join(
        f"{row.run} {row.variance_reduction:.2f}"
        if row.status == "ok"
        else f"{row.run} failed"
        for row in selection_df[~selection_df["selected"]].itertuples()
    )
    dropped = selection_df["dropped_stations"].dropna().astype(str)
    dropped = ", ".join(s for s in dropped if s) or "none"
    return (
        f"selected: {selected['run']} (VR {selected['variance_reduction']:.2f}; "
        f"{others}; dropped: {dropped}). "
        f"Mag: {solution['Mw']:.2f}; Depth: {solution['CD']:.2f} km; "
        f"Lat: {solution['Latitude']:.4f}; Lon: {solution['Longitude']:.4f}"
    )


def _read_cmt_timing_summary(output_dir: Path) -> str:
    """
    Build a short timing summary of a CMT pipeline for Slack.

    Parameters
    ----------
    output_dir : Path
        Directory the CMT pipeline wrote its results to.

    Returns
    -------
    str
        Total time, each run's time and the slowest stages across all runs
        (share of the summed run time), or an empty string if timings.csv is
        missing or can't be parsed.
    """
    try:
        timings_df = pd.read_csv(output_dir / TIMINGS_FILENAME)
    except (OSError, ValueError) as exc:
        print(
            f"[management.background] could not read CMT timings in {output_dir}: {exc}"
        )
        return ""

    is_total = timings_df["stage"] == "total"
    pipeline_total = timings_df.loc[
        is_total & (timings_df["run"] == "pipeline"), "seconds"
    ]
    run_totals = timings_df[is_total & (timings_df["run"] != "pipeline")]
    runs = ", ".join(
        f"{row.run} {format_duration(row.seconds)}" for row in run_totals.itertuples()
    )

    # Slowest stages summed over the runs
    stages_df = timings_df[
        (timings_df["run"] != "pipeline")
        & ~timings_df["stage"].isin(["total", "cleanup"])
    ]
    stage_seconds = (
        stages_df.groupby("stage")["seconds"].sum().nlargest(TIMING_TOP_STAGES)
    )
    run_seconds = max(run_totals["seconds"].sum(), 1e-9)
    stages = ", ".join(
        f"{stage} {100 * seconds / run_seconds:.0f}%"
        for stage, seconds in stage_seconds.items()
    )

    total = format_duration(pipeline_total.iloc[0]) if len(pipeline_total) else "?"
    return f"Timing: total {total} ({runs}); slowest stages: {stages}"


def _watch_and_notify(
    process: subprocess.Popen,
    event_id: str,
    output_dir: Path,
    log_path: Path,
    slack_thread_ts: str | None,
) -> None:
    """
    Wait for the CMT subprocess to exit and post a Slack reply reporting
    success or failure.

    Runs in a background daemon thread started by launch_cmt_background, so
    the ``process.wait()`` call here does not block the caller.

    Parameters
    ----------
    process : subprocess.Popen
        The running CMT subprocess.
    event_id : str
        GeoNet event/public ID.
    output_dir : Path
        Directory the CMT pipeline writes results to.
    log_path : Path
        Path to the subprocess's combined stdout/stderr log.
    slack_thread_ts : str, optional
        Slack thread timestamp to reply into. If None, no Slack message is
        sent.
    """
    returncode = process.wait()

    if slack_thread_ts is None:
        return

    if returncode == 0:
        message = (
            f"CMT inversion completed for Event ID: {event_id} "
            f"({_read_cmt_result_summary(output_dir)}). Results: {output_dir}\n"
            f"{_read_cmt_timing_summary(output_dir)}"
        )
    else:
        message = (
            f"CMT inversion FAILED for Event ID: {event_id}, "
            f"exit code {returncode}. Log: {log_path}\n"
            f"Last {LOG_TAIL_LINES} log lines:\n```{_tail(log_path)}```"
        )

    try:
        reply_to_message_on_slack(slack_thread_ts, message)
    except ValueError as exc:
        print(f"[management.background] failed to post CMT completion to Slack: {exc}")


def launch_cmt_background(
    event_id: str,
    event_csv_path: Path,
    output_dir: Path,
    nz_3dvm_path: Path | None = None,
    real_time: bool = True,
    python_executable: str = sys.executable,
    threads: int | None = None,
    slack_thread_ts: str | None = None,
    profile: bool = True,
) -> subprocess.Popen:
    """
    Launch the multi-run 1-D CMT pipeline for one event as a detached background process.

    The pipeline runs from ``nzgmdb.scripts.run_auto_cmt`` (all-stations,
    azimuth-control and outlier-dropped deviatoric runs, then selection of
    the best solution), so this function only launches and reports on it.
    The subprocess is started in its own session (``start_new_session=True``),
    so it keeps running even if the caller's process is later interrupted.
    A daemon thread waits on it in the background and, once it exits, posts
    a Slack reply reporting success or failure - this call itself never
    blocks or waits on the run.

    Parameters
    ----------
    event_id : str
        GeoNet event/public ID.
    event_csv_path : Path
        CSV with the event's finalised source parameters (evid, datetime,
        lat, lon, depth, mag, ...) - e.g. the same
        EARTHQUAKE_SOURCE_TABLE_GEONET flatfile NZGMDB already wrote/updated
        for this event.
    output_dir : Path
        Directory CMT results get written to (created if missing), e.g.
        ``event_dir / "cmt_1d"``.
    nz_3dvm_path : Path, optional
        3-D NZ velocity model CSV.
    real_time : bool, optional
        Passed through as ``--real-time`` - defaults to True since NZGMDB
        only calls this once an event has already been confirmed in
        near-real-time and may not have propagated to GeoNet's standard FDSN
        archive yet.
    python_executable : str, optional
        Interpreter to run the CMT pipeline with. Defaults to the interpreter
        running the caller.
    threads : int, optional
        Override the default Axitra thread count for each run.
    slack_thread_ts : str, optional
        Slack thread timestamp to reply into once the pipeline finishes. If
        omitted, no Slack notification is sent. On success, the message
        reports the selected run, each run's variance reduction, the dropped
        stations and the selected solution's magnitude and location, plus a
        timing summary.
    profile : bool, optional
        Write a cProfile dump for each run to <run>/results/profile.prof,
        by default True.

    Returns
    -------
    subprocess.Popen
        Handle to the detached process. Not waited on here - the caller may
        ignore it, or keep it to poll status later via ``.poll()``.
    """
    event_csv_path = Path(event_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_executable,
        "-m",
        CMT_MODULE,
        CMT_COMMAND,
        event_id,
        str(output_dir),
        "--event-csv",
        str(event_csv_path),
    ]
    if nz_3dvm_path:
        cmd += ["--nz-3dvm-path", str(nz_3dvm_path)]
    if real_time:
        cmd.append("--real-time")
    if threads is not None:
        cmd += ["--threads", str(threads)]
    if profile:
        cmd.append("--profile")

    log_path = output_dir / "cmt_1d_run.log"
    # The subprocess keeps its own handle to the log once started
    with open(log_path, "a") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    print(
        f"[management.background] launched CMT pipeline for {event_id} "
        f"(pid={process.pid}), logging to {log_path}"
    )

    watcher = threading.Thread(
        target=_watch_and_notify,
        args=(process, event_id, output_dir, log_path, slack_thread_ts),
        daemon=True,
    )
    watcher.start()

    return process
