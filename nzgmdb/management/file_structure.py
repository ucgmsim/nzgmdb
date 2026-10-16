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
    STATION_EXTRACTION_TABLE_GEONET = "station_extraction_table_geonet.csv"
    STATION_MAGNITUDE_TABLE_EXTRACTION = "station_magnitude_table_extraction.csv"
    MULTI_EVENT_TABLE = "multi_event_table.csv"
    EARTHQUAKE_SOURCE_TABLE_TECTONIC = "earthquake_source_table_tectonic.csv"
    EARTHQUAKE_SOURCE_TABLE_DISTANCES = "earthquake_source_table_distances.csv"
    EARTHQUAKE_SOURCE_TABLE_AFTERSHOCKS = "earthquake_source_table_aftershocks.csv"
    EARTHQUAKE_SOURCE_GEOMETRY = "earthquake_source_geometry_all.csv"
    PHASE_ARRIVAL_TABLE = "phase_arrival_table_all.csv"
    SITE_TABLE = "site_table_all.csv"
    STATION_TABLE = "station_table_all.csv"
    PROPAGATION_TABLE = "propagation_path_table_all.csv"
    GROUND_MOTION_IM_CATALOGUE = "ground_motion_im_catalogue.csv"
    PROB_SERIES = "prob_series.h5"


class FlatfileNames(StrEnum):
    """
    Final flatfile names for the NZGMDB results
    """

    EARTHQUAKE_SOURCE_TABLE = "earthquake_source_table.csv"
    EARTHQUAKE_SOURCE_GEOMETRY = "earthquake_source_geometry.csv"
    STATION_MAGNITUDE_TABLE = "station_magnitude_table.csv"
    STATION_EXTRACTION_TABLE = "station_extraction_table.csv"
    MULTI_EVENT_TABLE = "multi_event_table.csv"
    SITE_TABLE = "site_table.csv"
    STATION_TABLE = "station_table.csv"
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
    EXTRACTION_SKIPPED_RECORDS = "extraction_skipped_records.csv"
    MULTI_TRACE_ISSUE_RECORDS = "multi_trace_issue_records.csv"
    CLIPPED_RECORDS = "clipped_records.csv"
    IM_CALC_SKIPPED_RECORDS = "im_calc_skipped_records.csv"
    MISSING_SITES = "missing_sites.csv"
    PROCESSING_SKIPPED_RECORDS = "processing_skipped_records.csv"
    PHASE_ARRIVAL_SKIPPED_RECORDS = "phase_arrival_skipped_records.csv"
    SNR_SKIPPED_RECORDS = "snr_skipped_records.csv"
    FMAX_SKIPPED_RECORDS = "fmax_skipped_records.csv"
    GMC_SKIPPED_RECORDS = "gmc_skipped_records.csv"
    QUALITY_SKIPPED_RECORDS = "quality_skipped_records.csv"


def get_mseed_dir(main_dir: Path, year: int, event_id: str) -> Path:
    """
    Get the directory to save the mseed files.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory).
    year : int
        The year of the event.
    event_id : str
        The event ID.

    Returns
    -------
    Path
        The directory where the mseed files are stored.
    """
    return main_dir / "waveforms" / f"{year}" / event_id / "mseed"


def get_mseed_dir_from_snrfas(snrfas_file: Path) -> Path:
    """
    Get the mseed directory from an SNR/FAS file.

    Uses the snr_fas structure to determine the year and event ID.

    Parameters
    ----------
    snrfas_file : Path
        The snr_fas file.

    Returns
    -------
    Path
        The directory where the mseed files are stored.
    """
    main_dir = snrfas_file.parent.parent.parent.parent
    year = int(snrfas_file.parent.parent.name)
    event_id = snrfas_file.parent.name
    return get_mseed_dir(main_dir, year, event_id)


def get_snr_fas_ffp(snr_fas_dir: Path, mseed_file: Path) -> Path:
    """
    Get the SNR/FAS file path for an mseed file.

    Uses the same year and event ID folders as the mseed file (waveforms/<year>/<evid>/mseed).

    Parameters
    ----------
    snr_fas_dir : Path
        The directory where the SNR/FAS files are stored.
    mseed_file : Path
        The mseed file.

    Returns
    -------
    Path
        The SNR/FAS file path, named <record_id>_snr_fas.csv.
    """
    event_dir = mseed_file.parent.parent
    return (
        snr_fas_dir
        / event_dir.parent.name
        / event_dir.name
        / f"{mseed_file.stem}_snr_fas.csv"
    )


def get_processed_dir(main_dir: Path, year: int, event_id: str) -> Path:
    """
    Get the directory for processed waveform files.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results.
    year : int
        The year of the event.
    event_id : str
        The event ID.

    Returns
    -------
    Path
        The directory where processed waveforms are stored.
    """
    return main_dir / "waveforms" / f"{year}" / event_id / "processed"


def get_flatfile_dir(main_dir: Path) -> Path:
    """
    Get the directory for flatfiles.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results.

    Returns
    -------
    Path
        The directory where flatfiles are stored.
    """
    return main_dir / "flatfiles"


def get_quality_db_dir(main_dir: Path) -> Path:
    """
    Get the directory for the quality database.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results.

    Returns
    -------
    Path
        The directory where flatfiles in the quality database are stored.
    """
    return main_dir / "quality_db"


def get_event_id_from_mseed(mseed_file: Path) -> str:
    """
    Extract the event ID from an mseed file.

    Parameters
    ----------
    mseed_file : Path
        The mseed file.

    Returns
    -------
    str
        The event ID extracted from the file path.
    """
    return mseed_file.parent.parent.name


def get_config_dir() -> Path:
    """
    Get the directory where config files are stored.

    Returns
    -------
    Path
        The directory containing the config files.
    """
    return Path(__file__).parent.parent / "config"


def get_snr_fas_dir(main_dir: Path) -> Path:
    """
    Get the directory for storing SNR and FAS results.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results.

    Returns
    -------
    Path
        The directory where SNR and FAS results are stored.
    """
    return main_dir / "snr_fas"


def get_fmax_dir(main_dir: Path) -> Path:
    """
    Get the directory for storing the Fmax batch results.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results.

    Returns
    -------
    Path
        The directory where the Fmax batch results are stored.
    """
    return main_dir / "fmax"


def get_waveform_dir(main_dir: Path) -> Path:
    """
    Get the directory where waveforms (mseed and processed) are stored.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results.

    Returns
    -------
    Path
        The directory containing waveform files.
    """
    return main_dir / "waveforms"


def get_im_dir(main_dir: Path) -> Path:
    """
    Get the directory for storing IM results.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results.

    Returns
    -------
    Path
        The directory where IM results are stored.
    """
    return main_dir / "IM"


def get_processed_dir_from_mseed(mseed_file: Path) -> Path:
    """
    Get the directory for storing processed waveform files from an mseed file.

    Parameters
    ----------
    mseed_file : Path
        The mseed file.

    Returns
    -------
    Path
        The directory where processed files are stored.
    """
    return mseed_file.parent.parent / "processed"


def get_gmc_dir(main_dir: Path) -> Path:
    """
    Get the directory for storing GMC results.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results.

    Returns
    -------
    Path
        The directory where GMC results are stored.
    """
    return main_dir / "gmc"


def get_stationxml_dir(main_dir: Path) -> Path:
    """
    Get the directory for storing StationXML files.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results.

    Returns
    -------
    Path
        The directory where StationXML files are stored.
    """
    return main_dir / "stationxml"


def get_stationxml_ffp(xml_dir: Path, network: str, station: str) -> Path:
    """
    Get the StationXML file path for a station, named NET.STA.xml.

    Parameters
    ----------
    xml_dir : Path
        The directory where StationXML files are stored.
    network : str
        The network code of the station.
    station : str
        The station code.

    Returns
    -------
    Path
        The StationXML file path.
    """
    return xml_dir / f"{network}.{station}.xml"


def get_record_id(event_id: str, station: str, channel: str, location: str) -> str:
    """
    Get the record_id for a record, formatted as EVID_STA_CHAN_LOC.

    Parameters
    ----------
    event_id : str
        The event id.
    station : str
        The station code.
    channel : str
        The channel code, only the first 2 characters are used.
    location : str
        The location code, empty location codes are written as "00".

    Returns
    -------
    str
        The record_id.
    """
    # Empty location codes are written as "00" based on the FDSN Source Identifiers documentation
    # (https://docs.fdsn.org/projects/source-identifiers/en/latest/location-codes.html)
    return f"{event_id}_{station}_{channel[:2]}_{location or '00'}"
