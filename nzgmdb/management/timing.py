"""
Step timing and profiling helpers for tracking where pipeline time is spent.
"""

import cProfile
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

TIMINGS_FILENAME = "timings.csv"
PROFILE_FILENAME = "profile.prof"

# Forked worker processes would otherwise inherit an active profiler, slowing
# them down while their profile data is thrown away
os.register_at_fork(after_in_child=lambda: sys.setprofile(None))


class StepTimer:
    """
    Lap timer recording the time taken by each sequential pipeline step.

    Call ``lap(step)`` after each step finishes, the time since the
    previous lap (or creation) is recorded against that step.
    """

    def __init__(self):  # noqa: D107
        self.rows: list[dict] = []
        self._start = self._last = time.perf_counter()

    def lap(self, step: str) -> float:
        """
        Record the time since the previous lap against a step.

        Parameters
        ----------
        step : str
            Name of the step that just finished.

        Returns
        -------
        float
            Seconds taken by the step.
        """
        now = time.perf_counter()
        seconds = now - self._last
        self.rows.append({"step": step, "seconds": seconds})
        self._last = now
        return seconds

    def to_dataframe(self) -> pd.DataFrame:
        """
        Summarise the recorded steps.

        Returns
        -------
        pd.DataFrame
            One row per step plus "total", with seconds and percent of the total.
        """
        total = time.perf_counter() - self._start
        timings_df = pd.DataFrame(
            self.rows + [{"step": "total", "seconds": total}],
            columns=["step", "seconds"],
        )
        timings_df["percent"] = 100 * timings_df["seconds"] / max(total, 1e-9)
        return timings_df.round({"seconds": 2, "percent": 1})


def safe_write_timings(timings_df: pd.DataFrame, timings_ffp: Path) -> None:
    """
    Write a timings table, reporting rather than raising on failure.

    Used in ``finally`` blocks, so a failed write never hides the error that
    ended the run.

    Parameters
    ----------
    timings_df : pd.DataFrame
        Timings table to write.
    timings_ffp : Path
        CSV file to write to. Skipped if its folder no longer exists (e.g.
        a skipped event's directory was removed).
    """
    if not timings_ffp.parent.exists():
        return
    try:
        timings_df.to_csv(timings_ffp, index=False)
    except OSError as exc:
        print(f"[management.timing] could not write {timings_ffp}: {exc}")


@contextmanager
def profile_to(profile_ffp: Path | None) -> Iterator[None]:
    """
    Profile the code inside the context with cProfile and dump the stats.

    Only the calling process and thread are profiled, child processes are not.

    Parameters
    ----------
    profile_ffp : Path, optional
        File to write the cProfile stats to (view with snakeviz). If None,
        nothing is profiled.

    Yields
    ------
    None
        Profiling is active while inside the context.
    """
    if profile_ffp is None:
        yield
        return
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        # Never let a failed dump hide the error that ended the block
        if profile_ffp.parent.exists():
            try:
                profiler.dump_stats(profile_ffp)
            except OSError as exc:
                print(f"[management.timing] could not write {profile_ffp}: {exc}")


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a short human readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        e.g. "1h02m", "4m10s" or "12s".
    """
    seconds = round(float(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
