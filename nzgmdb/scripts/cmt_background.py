"""
Launch the near-real-time 1-D CMT inversion (run_cmt_1d) as a detached
background subprocess, and notify Slack when it finishes.
"""

import subprocess
import sys
import threading
from pathlib import Path

from nzgmdb.management.slack import reply_to_message_on_slack

CMT_MODULE = "nzgmdb.scripts.run_cmt_1d"
LOG_TAIL_LINES = 20


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


def _watch_and_notify(
    process: subprocess.Popen,
    event_id: str,
    output_dir: Path,
    log_path: Path,
    slack_thread_ts: str | None,
    mag: float | None,
    lat: float | None,
    lon: float | None,
    depth: float | None,
) -> None:
    """
    Wait for the CMT subprocess to exit and post a Slack reply reporting
    success or failure.

    Runs in a background daemon thread started by launch_cmt_background,
    so the ``process.wait()`` call here does not block the caller.

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
    mag : float, optional
        Event magnitude, included in the Slack message text.
    lat : float, optional
        Event latitude, included in the Slack message text.
    lon : float, optional
        Event longitude, included in the Slack message text.
    depth : float, optional
        Event depth (km), included in the Slack message text.
    """
    returncode = process.wait()

    if slack_thread_ts is None:
        return

    location = (
        f"Mag: {mag:.2f}; Depth: {depth:.2f} km; Lat: {lat:.4f}; Lon: {lon:.4f}"
        if None not in (mag, lat, lon, depth)
        else ""
    )

    if returncode == 0:
        message = (
            f"CMT inversion completed for Event ID: {event_id} ({location}). "
            f"Results: {output_dir}"
        )
    else:
        message = (
            f"CMT inversion FAILED for Event ID: {event_id} ({location}), "
            f"exit code {returncode}. Log: {log_path}\n"
            f"Last {LOG_TAIL_LINES} log lines:\n```{_tail(log_path)}```"
        )

    try:
        reply_to_message_on_slack(slack_thread_ts, message)
    except ValueError as exc:
        print(f"[cmt_background] failed to post CMT completion to Slack: {exc}")


def launch_cmt_background(
    event_id: str,
    event_csv_path: Path,
    output_dir: Path,
    *,
    python_executable: str = sys.executable,
    nz_3dvm_path: Path | None = None,
    threads: int | None = None,
    slack_thread_ts: str | None = None,
    mag: float | None = None,
    lat: float | None = None,
    lon: float | None = None,
    depth: float | None = None,
) -> subprocess.Popen:
    """
    Launch the 1-D CMT inversion for one event as a detached background process.

    The subprocess is started in its own session (``start_new_session=True``),
    so it keeps running even if the caller's process is later interrupted. A
    daemon thread waits on it in the background and, once it exits, posts a
    Slack reply reporting success or failure - this call itself never blocks
    or waits on the run.

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
    python_executable : str, optional
        Interpreter to run the CMT module with. Defaults to the interpreter
        running the caller, so it inherits whatever venv real_time_eq_runs.py
        is already running under (must have BayesISOLA installed).
    nz_3dvm_path : Path, optional
        Override the default 3-D velocity model CSV.
    threads : int, optional
        Override the default Axitra thread count.
    slack_thread_ts : str, optional
        Slack thread timestamp to reply into once the run finishes. If
        omitted, no Slack notification is sent.
    mag : float, optional
        Event magnitude, included in the Slack notification text.
    lat : float, optional
        Event latitude, included in the Slack notification text.
    lon : float, optional
        Event longitude, included in the Slack notification text.
    depth : float, optional
        Event depth (km), included in the Slack notification text.

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
    if nz_3dvm_path is not None:
        cmd += ["--nz-3dvm-path", str(nz_3dvm_path)]
    if threads is not None:
        cmd += ["--threads", str(threads)]

    log_path = output_dir / "cmt_1d_run.log"
    log_file = open(log_path, "a")

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # detach from this process's session/group
    )
    log_file.close()  # the child keeps its own duplicated file descriptor

    print(f"[cmt_background] launched CMT run for {event_id} "
          f"(pid={process.pid}), logging to {log_path}")

    watcher = threading.Thread(
        target=_watch_and_notify,
        args=(process, event_id, output_dir, log_path, slack_thread_ts, mag, lat, lon, depth),
        daemon=True,
    )
    watcher.start()

    return process
