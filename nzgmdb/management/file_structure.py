"""
File containing the functions for organizing the different files into a proper structure.
For both the NZGMDB results and the data files used to generate the results within this repository.
"""

from enum import StrEnum
from pathlib import Path


class PreFlatfileNames(StrEnum):
    """
    Filenames of files that keep track of progress through the NZGMDB pipeline.
    Allows to easily return to a previous state if the pipeline fails.
    """

    EARTHQUAKE_SOURCE_TABLE_GEONET = "earthquake_source_table_geonet.csv"
    STATION_MAGNITUDE_TABLE_GEONET = "station_magnitude_table_geonet.csv"
    EARTHQUAKE_SOURCE_TABLE_TECTONIC = "earthquake_source_table_tectonic.csv"
    EARTHQUAKE_SOURCE_TABLE_DISTANCES = "earthquake_source_table_distances.csv"
    PHASE_ARRIVAL_TABLE = "phase_arrival_table_all.csv"
    SITE_TABLE = "site_table_all.csv"
    PROPAGATION_TABLE = "propagation_path_table_all.csv"
    GROUND_MOTION_IM_CATALOGUE = "ground_motion_im_catalogue.csv"
    PROB_SERIES = "prob_series.h5"


class FlatfileNames(StrEnum):
    """
    Final flatfile names for the NZGMDB results
    """

    EARTHQUAKE_SOURCE_TABLE = "earthquake_source_table.csv"
    STATION_MAGNITUDE_TABLE = "station_magnitude_table.csv"
    SITE_TABLE = "site_table.csv"
    PHASE_ARRIVAL_TABLE = "phase_arrival_table.csv"
    PROPAGATION_TABLE = "propagation_path_table.csv"
    GMC_PREDICTIONS = "gmc_predictions.csv"
    FMAX = "fmax.csv"
    SNR_METADATA = "snr_metadata.csv"
    GROUND_MOTION_IM_000_FLAT = "ground_motion_im_table_000_flat.csv"
    GROUND_MOTION_IM_090_FLAT = "ground_motion_im_table_090_flat.csv"
    GROUND_MOTION_IM_VER_FLAT = "ground_motion_im_table_ver_flat.csv"
    GROUND_MOTION_IM_GEOM_FLAT = "ground_motion_im_table_geom_flat.csv"
    GROUND_MOTION_IM_ROTD0_FLAT = "ground_motion_im_table_rotd0_flat.csv"
    GROUND_MOTION_IM_ROTD50_FLAT = "ground_motion_im_table_rotd50_flat.csv"
    GROUND_MOTION_IM_ROTD100_FLAT = "ground_motion_im_table_rotd100_flat.csv"
    GROUND_MOTION_IM_EAS_FLAT = "ground_motion_im_table_eas_flat.csv"


class SkippedRecordFilenames(StrEnum):
    """
    Filenames of files that keep track of records that were skipped during the NZGMDB pipeline.
    """

    GEONET_SKIPPED_RECORDS = "geonet_skipped_records.csv"
    CLIPPED_RECORDS = "clipped_records.csv"
    IM_CALC_SKIPPED_RECORDS = "im_calc_skipped_records.csv"
    MISSING_SITES = "missing_sites.csv"
    PROCESSING_SKIPPED_RECORDS = "processing_skipped_records.csv"
    PHASE_ARRIVAL_SKIPPED_RECORDS = "phase_arrival_skipped_records.csv"
    SNR_SKIPPED_RECORDS = "snr_skipped_records.csv"
    FMAX_SKIPPED_RECORDS = "fmax_skipped_records.csv"
    QUALITY_SKIPPED_RECORDS = "quality_skipped_records.csv"


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


def get_mseed_dir_from_snrfas(snrfas_file: Path):
    """
    Get the mseed directory of the current snr_fas record file
    (Uses the snr_fas structure to get the year and event id / main_dir)

    Parameters
    ----------
    snrfas_file : Path
        The snr_fas file
    """
    main_dir = snrfas_file.parent.parent.parent.parent
    year = int(snrfas_file.parent.parent.name)
    event_id = snrfas_file.parent.name
    mseed_dir = get_mseed_dir(main_dir, year, event_id)
    return mseed_dir


def get_processed_dir(main_dir: Path, year: int, event_id: str):
    """
    Get the directory to the processed waveform files
    given the year and event id

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


def get_waveform_dir(main_dir: Path):
    """
    Get the directory to where the waveforms are stored,
    both mseed and processed files

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    """
    waveform_dir = main_dir / "waveforms"
    return waveform_dir


def get_im_dir(main_dir: Path):
    """
    Get the directory to save the IM results

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    """
    im_dir = main_dir / "IM"
    return im_dir


def get_processed_dir_from_mseed(mseed_file: Path):
    """
    Get the directory to save the processed files

    Parameters
    ----------
    mseed_file : Path
        The mseed file

    Returns
    -------
    Path
        The directory to save the processed files
    """
    return mseed_file.parent.parent / "processed"


def get_gmc_dir(main_dir: Path):
    """
    Get the directory to save the GMC results

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    """
    gmc_dir = main_dir / "gmc"
    return gmc_dir
