"""
Launch the 1-D CMT inversion as a detached background subprocess, and notify
Slack when it finishes.
"""

import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

from nzgmdb.management.slack import reply_to_message_on_slack

CMT_MODULE = "auto_cmt.run_cmt"
LOG_TAIL_LINES = 20

# Written by auto_cmt.run_cmt into its output_dir once a run completes.
CMT_SOLUTION_FILENAME = "cmt_solution.csv"


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


def _read_cmt_result_location(output_dir: Path) -> tuple[float, float, float, float] | None:
    """
    Read the moment magnitude, latitude, longitude, and centroid depth from
    a completed CMT run's cmt_solution.csv.

    Parameters
    ----------
    output_dir : Path
        Directory the CMT run wrote its results to.

    Returns
    -------
    tuple of float, optional
        ``(mag, lat, lon, depth)`` read from cmt_solution.csv's ``Mw``,
        ``Latitude``, ``Longitude``, and ``CD`` columns, or None if the file
        is missing or can't be parsed.
    """
    cmt_solution_path = output_dir / CMT_SOLUTION_FILENAME
    try:
        row = pd.read_csv(cmt_solution_path).iloc[0]
        return (
            float(row["Mw"]),
            float(row["Latitude"]),
            float(row["Longitude"]),
            float(row["CD"]),
        )
    except (OSError, KeyError, IndexError, ValueError) as exc:
        print(f"[management.background] could not read {cmt_solution_path}: {exc}")
        return None


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
        Directory the CMT run writes results to.
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
        result = _read_cmt_result_location(output_dir)
        if result is not None:
            mag, lat, lon, depth = result
            location = f"Mag: {mag:.2f}; Depth: {depth:.2f} km; Lat: {lat:.4f}; Lon: {lon:.4f}"
        else:
            location = ""
        message = (
            f"CMT inversion completed for Event ID: {event_id} ({location}). "
            f"Results: {output_dir}"
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
) -> subprocess.Popen:
    """
    Launch the 1-D CMT inversion for one event as a detached background process.

    The inversion itself runs from the ``cmt_solutions`` package's
    ``auto_cmt.run_cmt`` module (``python -m auto_cmt.run_cmt``), so this
    function only launches and reports on it. The subprocess is started in
    its own session (``start_new_session=True``), so it keeps running even
    if the caller's process is later interrupted. A daemon thread waits on
    it in the background and, once it exits, posts a Slack reply reporting
    success or failure - this call itself never blocks or waits on the run.

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
        Directory CMT results/figures/raw data get written to (created if
        missing), e.g. ``event_dir / "cmt_1d"``.
    nz_3dvm_path : Path, optional
        3-D NZ velocity model CSV.
    real_time : bool, optional
        Passed through as ``auto_cmt.run_cmt --real-time`` - defaults to
        True since NZGMDB only calls this once an event has already been
        confirmed in near-real-time and may not have propagated to GeoNet's
        standard FDSN archive yet.
    python_executable : str, optional
        Interpreter to run the CMT module with. Defaults to the interpreter
        running the caller.
    threads : int, optional
        Override the default Axitra thread count.
    slack_thread_ts : str, optional
        Slack thread timestamp to reply into once the run finishes. If
        omitted, no Slack notification is sent. On success, the magnitude,
        latitude, longitude, and depth included in that message are read
        back from the run's own cmt_solution.csv.

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
        python_executable, "-m", CMT_MODULE,
        event_id, str(event_csv_path), str(output_dir),
    ]
    if nz_3dvm_path:
        cmd += ["--nz-3dvm-path", str(nz_3dvm_path),]
    if real_time:
        cmd.append("--real-time")
    if threads is not None:
        cmd += ["--threads", str(threads)]

    log_path = output_dir / "cmt_1d_run.log"
    log_file = open(log_path, "a")

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log_file.close()

    print(f"[management.background] launched CMT run for {event_id} "
          f"(pid={process.pid}), logging to {log_path}")

    watcher = threading.Thread(
        target=_watch_and_notify,
        args=(process, event_id, output_dir, log_path, slack_thread_ts),
        daemon=True,
    )
    watcher.start()

    return process
