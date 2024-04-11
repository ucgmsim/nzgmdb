"""
File containing the functions for organizing the different files into a proper structure.
"""

from pathlib import Path


def get_mseed_dir(main_dir: Path, year: int, event_id: str):
    """
    Get the directory to save the mseed files

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    year : int
        The year of the event
    event_id : str
        The event id
    """
    mseed_dir = main_dir / "waveforms" / f"{year}" / event_id / "mseed"
    return mseed_dir


def get_processed_dir(main_dir: Path, year: int, event_id: str):
    """
    Get the directory to save the processed waveform files

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    year : int
        The year of the event
    event_id : str
        The event id
    """
    processed_dir = main_dir / "waveforms" / f"{year}" / event_id / "processed"
    return processed_dir


def get_flatfile_dir(main_dir: Path):
    """
    Get the directory to the flatfiles

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    """
    flatfile_dir = main_dir / "flatfiles"
    return flatfile_dir
