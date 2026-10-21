"""
Clean up BayesISOLA CMT run folders, keeping only results, figures, reports and logs.
"""

import os
import shutil
from pathlib import Path

# Matches auto_cmt.run_cmt.CMT_SOLUTION_FILENAME, kept as a plain string so
# importing this module doesn't import BayesISOLA
CMT_SOLUTION_FILENAME = "cmt_solution.csv"

# Files/folders kept in a run folder by cleanup_cmt_run, plus any *.log file
CLEANUP_KEEP = {
    "results",
    "figures",
    "html",
    CMT_SOLUTION_FILENAME,
    "report.html",
    "index.html",
    "log.txt",
    "log_green.txt",
}


def find_cmt_run_dirs(path: Path) -> list[Path]:
    """
    Find all CMT run folders at or below a path.

    A run folder is any folder containing results/solution_summary.csv.

    Parameters
    ----------
    path : Path
        A run folder, an event's CMT folder or a folder of events.

    Returns
    -------
    list[Path]
        Sorted run folders.
    """
    run_dirs = []
    for dirpath, dirnames, _ in os.walk(path):
        if (Path(dirpath) / "results" / "solution_summary.csv").is_file():
            run_dirs.append(Path(dirpath))
            # Don't descend into a run folder (e.g. green/ is huge)
            dirnames[:] = []
    return sorted(run_dirs)


def _path_size(path: Path) -> int:
    """
    Get the size in bytes of a file or folder.

    Parameters
    ----------
    path : Path
        File or folder.

    Returns
    -------
    int
        Total size in bytes.
    """
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(f.lstat().st_size for f in path.rglob("*") if not f.is_dir())


def cleanup_cmt_run(run_dir: Path, dry_run: bool = False) -> int:
    """
    Delete everything in a CMT run folder except the files in CLEANUP_KEEP and logs.

    Parameters
    ----------
    run_dir : Path
        CMT run folder, must contain a results/ folder.
    dry_run : bool, optional
        Only print what would be deleted, by default False.

    Returns
    -------
    int
        Bytes deleted (or that would be deleted for a dry run).
    """
    if not (run_dir / "results").is_dir():
        raise ValueError(f"{run_dir} does not look like a CMT run folder (no results/)")

    freed = 0
    for path in sorted(run_dir.iterdir()):
        if path.name in CLEANUP_KEEP or path.suffix == ".log":
            continue
        size = _path_size(path)
        freed += size
        if dry_run:
            print(f"[dry-run] would delete {path} ({size / 1e6:.1f} MB)")
        elif path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    return freed
