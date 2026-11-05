"""
Clean up BayesISOLA CMT run folders, keeping only results, figures, reports and logs,
and delete the Green's function stores shared by the runs of an event.
"""

import os
import shutil
from pathlib import Path

# Matches auto_cmt.run_cmt.CMT_SOLUTION_FILENAME, kept as a plain string so
# importing this module doesn't import BayesISOLA
CMT_SOLUTION_FILENAME = "cmt_solution.csv"

# Green's function store shared by the runs of an event (in the event's CMT
# folder), only needed while the runs are going
GF_STORE_DIRNAME = "gf_store"

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
        else:
            dirnames[:] = [d for d in dirnames if d != GF_STORE_DIRNAME]
    return sorted(run_dirs)


def find_gf_store_dirs(path: Path) -> list[Path]:
    """
    Find all shared Green's function stores at or below a path.

    Parameters
    ----------
    path : Path
        An event's CMT folder or a folder of events.

    Returns
    -------
    list[Path]
        Sorted gf_store folders.
    """
    gf_store_dirs = []
    for dirpath, dirnames, _ in os.walk(path):
        # Don't descend into run folders (their own green/ is cleaned with the run)
        if (Path(dirpath) / "results" / "solution_summary.csv").is_file():
            dirnames[:] = []
            continue
        if GF_STORE_DIRNAME in dirnames:
            gf_store_dirs.append(Path(dirpath) / GF_STORE_DIRNAME)
            dirnames.remove(GF_STORE_DIRNAME)
    return sorted(gf_store_dirs)


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


def cleanup_gf_store(gf_store_dir: Path, dry_run: bool = False) -> int:
    """
    Delete a shared Green's function store.

    Parameters
    ----------
    gf_store_dir : Path
        gf_store folder of an event's CMT folder.
    dry_run : bool, optional
        Only print what would be deleted, by default False.

    Returns
    -------
    int
        Bytes deleted (or that would be deleted for a dry run).
    """
    if gf_store_dir.name != GF_STORE_DIRNAME:
        raise ValueError(f"{gf_store_dir} is not a {GF_STORE_DIRNAME} folder")
    size = _path_size(gf_store_dir)
    if dry_run:
        print(f"[dry-run] would delete {gf_store_dir} ({size / 1e6:.1f} MB)")
    else:
        shutil.rmtree(gf_store_dir)
    return size
