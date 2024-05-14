"""
File containing the functions for organizing the different files into a proper structure.
For both the NZGMDB results and the data files used to generate the results within this repository.
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


def get_event_id_from_mseed(mseed_file: Path):
    """
    Get the event id from the mseed file

    Parameters
    ----------
    mseed_file : Path
        The mseed file
    """
    event_id = mseed_file.parent.parent.name
    return event_id


def get_data_dir():
    """
    Get the directory to the data files
    """
    return Path(__file__).parent.parent / "data"


def get_snr_fas_dir(main_dir: Path):
    """
    Get the directory to save the SNR and FAS results

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    """
    snr_fas_dir = main_dir / "snr_fas"
    return snr_fas_dir
