"""
Clean up CMT run folders inside a folder of live events, e.g. for runs made before automatic cleanup.
"""

from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.management.cmt_cleanup import cleanup_cmt_run, find_cmt_run_dirs
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


def find_event_cmt_dirs(
    live_events_dir: Path, event_ids: list[str] | None = None
) -> list[Path]:
    """
    Find folders with "cmt" in their name directly inside each event folder.

    Parameters
    ----------
    live_events_dir : Path
        Folder of live events, one sub folder per event ID.
    event_ids : list[str], optional
        Only check these events, by default all events.

    Returns
    -------
    list[Path]
        Sorted CMT folders, e.g. <event_id>/cmt_1d and <event_id>/cmt_1d_deviatoric.
    """
    event_dirs = (
        [live_events_dir / event_id for event_id in event_ids]
        if event_ids
        else [path for path in live_events_dir.iterdir() if path.is_dir()]
    )
    return sorted(
        path
        for event_dir in event_dirs
        if event_dir.is_dir()
        for path in event_dir.iterdir()
        if path.is_dir() and "cmt" in path.name.lower()
    )


@cli.from_docstring(app)
def cleanup_old_cmts(
    live_events_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    event_ids: Annotated[list[str] | None, typer.Option()] = None,
    dry_run: Annotated[bool, typer.Option(is_flag=True)] = False,
) -> None:
    """
    Clean up every CMT run inside the CMT folders of a folder of live events.

    Looks in each event folder for sub folders with "cmt" in their name, then
    cleans every completed run folder (containing results/solution_summary.csv)
    found in them. CMT folders without a completed run (e.g. failed runs) are
    listed but left untouched.

    Parameters
    ----------
    live_events_dir : Path
        Folder of live events, one sub folder per event ID.
    event_ids : list[str], optional
        Only check these events (repeatable), by default all events.
    dry_run : bool, optional
        Only print what would be deleted.
    """
    cmt_dirs = find_event_cmt_dirs(live_events_dir, event_ids)
    total = 0
    n_runs = 0
    no_runs = []
    for cmt_dir in cmt_dirs:
        run_dirs = find_cmt_run_dirs(cmt_dir)
        if not run_dirs:
            no_runs.append(cmt_dir)
            continue
        for run_dir in run_dirs:
            freed = cleanup_cmt_run(run_dir, dry_run=dry_run)
            total += freed
            n_runs += 1
            print(f"{run_dir.relative_to(live_events_dir)}: {freed / 1e9:.2f} GB")

    if no_runs:
        print(f"\n{len(no_runs)} CMT folder(s) with no completed run, left untouched:")
        for cmt_dir in no_runs:
            print(f"  {cmt_dir.relative_to(live_events_dir)}")

    action = "would free" if dry_run else "freed"
    print(
        f"\n{len(cmt_dirs)} CMT folder(s), {n_runs} run(s) cleaned, "
        f"{action} {total / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    app()
