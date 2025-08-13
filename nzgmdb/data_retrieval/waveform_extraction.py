"""
Holds all the functions for extracting waveforms for the NZGMDB database from the FDSN Client.
"""

import functools
import http.client
import multiprocessing as mp
import time
import warnings
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import (
    FDSNNoDataException,
    FDSNServiceUnavailableException,
)
from obspy.io.mseed import InternalMSEEDError, ObsPyMSEEDFilesizeTooSmallError
from pandas.errors import EmptyDataError

from nzgmdb.data_processing import filtering
from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors, file_structure
from nzgmdb.mseed_management import creation


def get_inital_stream(
    start_time: datetime,
    end_time: datetime,
    channel_codes: str,
    location: str,
    client: FDSN_Client,
    net: str,
    sta: str,
):
    """
    Get the initial stream of waveforms from the FDSN client with multiple retries for incomplete reads.

    Parameters
    ----------
    start_time : datetime
        The start time of the waveform data to retrieve.
    end_time : datetime
        The end time of the waveform data to retrieve.
    channel_codes : str
        The channel codes to retrieve, formatted as a comma-separated string.
        e.g. "HN?,BN?,HH?".
    location : str
        The location code to retrieve waveforms for, typically "*".
    client : FDSN_Client
        The FDSN client to use for retrieving waveforms.
    net : str
        The network code to retrieve waveforms for.
    sta : str
        The station code to retrieve waveforms for.

    Returns
    -------
    Stream
        An ObsPy Stream object containing the waveform data for the specified parameters.
    """
    # Get the waveforms with multiple retries when IncompleteReadError occurs
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                st = client.get_waveforms(
                    net,
                    sta,
                    location,
                    channel_codes,
                    start_time,
                    end_time,
                    attach_response=True,
                )
            break
        except FDSNNoDataException:
            return None
        except ObsPyMSEEDFilesizeTooSmallError:
            return None
        except (http.client.IncompleteRead, InternalMSEEDError):
            if attempt < max_retries - 1:  # i.e. not the last attempt
                continue  # try again
            else:
                return None
        except FDSNServiceUnavailableException:
            print(f"Error getting waveforms for {net}.{sta}")
            print("Service temporarily unavailable")
            print("HTTP Status code: 503")
            print("Retrying in 2 minutes...")
            time.sleep(120)  # Wait for 2 minutes before retrying
        except Exception as e:  # noqa: BLE001
            print(f"Unexpected error getting waveforms for {net}.{sta}")
            print(e)
            return None
    return st


def split_stream_into_mseeds(st: Stream, unique_channels: Iterable, event_id: str):
    """
    Split the stream object into multiple mseed files based on the unique channel and location

    Parameters
    ----------
    st : Stream
        The stream object containing the waveform data for every channel and location
    unique_channels : Iterable
        An Iterable of tuples containing the unique channel and location for each mseed file created
        [(channel, location), ...]
    event_id : str
        The event id which is used if there is a raised issue with the mseed file

    Returns
    -------
    list
        A list of stream objects containing the waveform data for each mseed file created
    """
    mseeds = []
    raised_issues = []
    for chan, loc in unique_channels:
        # Each unique channel and location pair is a new mseed file
        st_new = st.select(location=loc, channel=f"{chan}?")
        record_id = f"{event_id}_{st_new[0].stats.station}_{st_new[0].stats.channel[:2]}_{st_new[0].stats.location}"

        if len(st_new) > 3:
            # Check if all the sample rates are the same
            samples = [tr.stats.sampling_rate for tr in st_new]
            if len(set(samples)) > 1:
                # If they are different take the highest and resample with the others using interpolation
                st_new = st_new.select(sampling_rate=max(samples))
                st_new.merge(fill_value="interpolate")
                raised_issues.append(
                    [record_id, "Split stream, different sample rates"]
                )

        # Check the final length of the traces
        if len(st_new) != 3:
            raised_issues.append([record_id, "Unknown issue, multiple traces"])
            continue

        # Ensure traces all have the same length
        starttime_trim = max([tr.stats.starttime for tr in st_new])
        endtime_trim = min([tr.stats.endtime for tr in st_new])
        # Check that the start time is before the end time
        if starttime_trim > endtime_trim:
            raised_issues.append(
                [record_id, "Unknown issue, start time after end time"]
            )
            continue
        st_new.trim(starttime_trim, endtime_trim)

        mseeds.append(st_new)

    return mseeds, raised_issues


def get_station_window(
    station_extraction_row: pd.Series,
    client: FDSN_Client,
    channel_codes: str,
    loc: str,
):
    """
    Get the initial stream of waveforms for a station based on the extraction parameters.

    Parameters
    ----------
    station_extraction_row : pd.Series
        A row from the station extraction table containing the parameters for waveform extraction.
    client : FDSN_Client
        The FDSN client to use for retrieving waveforms.
    channel_codes : str
        The channel codes to retrieve, formatted as a comma-separated string.
        e.g. "HN?,BN?,HH?".
    loc : str
        The location code to retrieve waveforms for, typically "*".

    Returns
    -------
    Stream
        An ObsPy Stream object containing the waveform data for the specified station.
    """
    # Extract the parameters from the row
    net = station_extraction_row["net"]
    sta = station_extraction_row["sta"]
    ptime_est = station_extraction_row["ptime_est"]
    ds_mean = station_extraction_row["ds_mean"]
    ds_std = station_extraction_row["ds_std"]

    # Get the config values
    config = cfg.Config()
    pre_event_time_difference = config.get_value("pre_event_time_difference")

    start_time = ptime_est - pre_event_time_difference
    # Note: both ds_mean and ds_std are in logspace
    end_time = ptime_est + np.exp(ds_mean) * np.exp(3 * ds_std)

    return get_inital_stream(start_time, end_time, channel_codes, loc, client, net, sta)


def extract_station_info(
    station_extraction_row: pd.Series,
    main_dir: Path,
    client: FDSN_Client,
    only_record_ids: pd.DataFrame = None,
):
    """
    Extract the waveform data for a single station based on the extraction parameters.

    Parameters
    ----------
    station_extraction_row : pd.Series
        A row from the station extraction table containing the parameters for waveform extraction.
    main_dir : Path
        The main directory of the NZGMDB results (Highest Level Directory).
    client : FDSN_Client
        The FDSN client to use for retrieving waveforms.
    only_record_ids : pd.DataFrame, optional
        A DataFrame containing a subset of record IDs to use for extraction, if provided.

    Returns
    -------
    list
        A list of lists containing the station magnitude data.
    list
        A list of lists containing the skipped records.
    list
        A list of lists containing the clipped records.
    """
    sta_mag_line, skipped_records, clipped_records = [], [], []
    # Extract the parameters from the row
    event_id = station_extraction_row["evid"]
    station = station_extraction_row["sta"]
    network = station_extraction_row["net"]
    mag = station_extraction_row["mag"]
    pref_mag_type = station_extraction_row["pref_mag_type"]
    r_hyp = station_extraction_row["r_hyp"]

    # Get the catalogue information
    cat = client.get_events(eventid=event_id)
    event_cat = cat[0]

    # Obtain the station channel codes and location
    config = cfg.Config()
    channel_codes = ",".join(config.get_value("channel_codes"))
    location = "*"

    # Check what channel codes and locations to use from only_record_ids if provided
    if only_record_ids is not None:
        site_only_record_ids = only_record_ids[
            only_record_ids["record_id"].str.contains(f"_{station}_")
        ]
        # Get the channel and location to use
        channel_codes = (
            site_only_record_ids["record_id"].str.split("_").str[-2].values[0] + "?"
        )
        location = site_only_record_ids["record_id"].str.split("_").str[-1].values[0]

    # Get the Stream
    st = get_station_window(station_extraction_row, client, channel_codes, location)

    # Check that data was found
    if st is None:
        skipped_records.append([f"{event_id}_{station}", "No Waveform Data"])
        return sta_mag_line, skipped_records, clipped_records

    # Get the unique channels (Using first 2 keys) and locations
    unique_channels = set([(tr.stats.channel[:2], tr.stats.location) for tr in st])

    # Split the stream into mseeds
    mseeds, raised_issues = split_stream_into_mseeds(st, unique_channels, event_id)

    # Extend the raised_issues list with the skipped records
    skipped_records.extend(raised_issues)

    # Get the station magnitudes
    station_magnitudes = [
        mag
        for mag in event_cat.station_magnitudes
        if mag.waveform_id.station_code == station.code
    ]

    for mseed in mseeds:
        try:
            # Check the data is not all 0's
            if all([np.allclose(tr.data, 0) for tr in mseed]):
                stats = mseed[0].stats
                skipped_records.append(
                    [
                        f"{event_id}_{stats.station}_{stats.channel}_{stats.location}",
                        "All 0's",
                    ]
                )
                continue
        except TypeError:
            stats = mseed[0].stats
            skipped_records.append(
                [
                    f"{event_id}_{stats.station}_{stats.channel}_{stats.location}",
                    "TypeError when checking for all 0's",
                ]
            )

        # Calculate clip to determine if the record should be dropped
        clip = filtering.get_clip_probability(mag, r_hyp, mseed)

        threshold = config.get_value("clip_threshold")

        # Check if the record should be dropped
        if clip > threshold:
            stats = mseed[0].stats
            clipped_records.append(
                [
                    f"{event_id}_{stats.station}_{stats.channel[:2]}_{stats.location}",
                    "Clipped",
                ]
            )

        # Create the directory structure for the given event
        year = event_cat.origins[0].time.year
        mseed_dir = file_structure.get_mseed_dir(main_dir, year, event_id)

        # Write the mseed file
        creation.write_mseed(mseed, event_id, station, mseed_dir)

        for trace in mseed:
            chan = trace.stats.channel
            loc = trace.stats.location
            # Find the station magnitude
            # Ensures that the station codes matches and that if the channel code ends with Z then it makes
            # sure that the station magnitude is for the Z channel, otherwise any that match with the first two
            # characters of the channel code is sufficient
            sta_mag = None
            for mag in station_magnitudes:
                if mag.waveform_id.channel_code[:2] == chan[:2]:
                    sta_mag = mag
                    if chan[-1] == "Z":
                        break

            if sta_mag:
                sta_mag_mag = sta_mag.mag
                sta_mag_type = sta_mag.station_magnitude_type
                amp = next(
                    (
                        amp
                        for amp in event_cat.amplitudes
                        if amp.resource_id == sta_mag.amplitude_id
                    ),
                    None,
                )
            else:
                sta_mag_mag = None
                sta_mag_type = pref_mag_type
                amp = None

            # Get the amp values
            amp_amp = amp.generic_amplitude if amp else None
            amp_unit = amp.unit if amp and "unit" in amp else None

            mag_id = f"{event_id}m{len(sta_mag_line) + 1}"
            sta_mag_line.append(
                [
                    mag_id,
                    network,
                    station,
                    loc,
                    chan,
                    event_id,
                    sta_mag_mag,
                    sta_mag_type,
                    "uncorrected",
                    amp_amp,
                    amp_unit,
                ]
            )

    return sta_mag_line, skipped_records, clipped_records


def extract_waveforms(
    main_dir: Path,
    station_extraction_table_ffp: Path,
    n_procs: int = 1,
    only_record_ids_ffp: Path = None,
    batch_size: int = 1000,
):
    """
    Extract waveforms for each station in the station extraction table.
    Saves the results to the waveform directory and creates the
    station magnitude table, skipped records, and clipped records files.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest Level Directory).
    station_extraction_table_ffp : Path
        Path to the station extraction table CSV file.
        This file should contain the parameters for waveform extraction.
    n_procs : int, optional
        The number of processes to use for parallel extraction, by default 1.
    only_record_ids_ffp : Path, optional
        Full file path to the file containing a subset of record IDs to use for extraction, if provided.
    batch_size : int, optional
        The number of rows to process in each batch, by default 1000.
    """
    client_NZ = FDSN_Client("GEONET")
    station_extraction_table = pd.read_csv(
        station_extraction_table_ffp, dtype={"evid": str}
    )
    # Convert the p_time_est column to datetime
    station_extraction_table["ptime_est"] = pd.to_datetime(
        station_extraction_table["ptime_est"]
    )
    only_record_ids = (
        None if only_record_ids_ffp is None else pd.read_csv(only_record_ids_ffp)
    )

    # Get the batch directory
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    batch_dir = flatfile_dir / "extraction_batch_files"
    batch_dir.mkdir(exist_ok=True, parents=True)

    # Find files that have already been processed and get the suffix indexes and remove them from the event_ids
    processed_files = [f for f in batch_dir.iterdir() if f.is_file()]
    processed_suffixes = set(int(f.stem.split("_")[-1]) for f in processed_files)

    # Split the DataFrame index into batches
    index_batches = np.array_split(
        station_extraction_table.index,
        np.ceil(len(station_extraction_table) / batch_size),
    )

    for batch_index, batch_indices in enumerate(index_batches):
        if batch_index not in processed_suffixes:
            batch_rows = station_extraction_table.loc[batch_indices]
            with mp.Pool(n_procs) as pool:
                results = pool.map(
                    functools.partial(
                        extract_station_info,
                        main_dir=main_dir,
                        client=client_NZ,
                        only_record_ids=only_record_ids,
                    ),
                    (row for _, row in batch_rows.iterrows()),
                )

            # Extract the results
            sta_mag_data, skipped_records, clipped_records = [], [], []
            for result in results:
                (
                    finished_sta_mag_data,
                    finished_skipped_records,
                    finished_clipped_records,
                ) = result
                sta_mag_data.extend(finished_sta_mag_data)
                skipped_records.extend(finished_skipped_records)
                clipped_records.extend(finished_clipped_records)

            if len(sta_mag_data) > 0:
                sta_mag_df = pd.DataFrame(
                    sta_mag_data,
                    columns=[
                        "magid",
                        "net",
                        "sta",
                        "loc",
                        "chan",
                        "evid",
                        "mag",
                        "mag_type",
                        "mag_corr_method",
                        "amp",
                        "amp_unit",
                    ],
                )
            else:
                sta_mag_df = pd.DataFrame()

            sta_mag_df.to_csv(
                batch_dir / f"station_magnitude_table_{batch_index}.csv", index=False
            )

            if len(skipped_records) > 0:
                # Create the skipped records df
                skipped_records_df = pd.DataFrame(
                    skipped_records, columns=["skipped_records", "reason"]
                )
            else:
                skipped_records_df = pd.DataFrame()

            skipped_records_df.to_csv(
                batch_dir / f"extraction_skipped_records_{batch_index}.csv", index=False
            )

            if len(clipped_records) > 0:
                # Create the clipped records df
                clipped_records_df = pd.DataFrame(
                    clipped_records, columns=["record_id", "reason"]
                )
            else:
                clipped_records_df = pd.DataFrame()

            clipped_records_df.to_csv(
                batch_dir / f"extraction_clipped_records_{batch_index}.csv", index=False
            )

    # Combine all the event and sta_mag dataframes
    sta_mag_dfs = []
    skipped_records_dfs = []
    clipped_records_dfs = []

    for file in batch_dir.iterdir():
        if "station_magnitude_table" in file.stem:
            try:
                sta_mag_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")
        elif "extraction_skipped_records" in file.stem:
            try:
                skipped_records_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")
        elif "extraction_clipped_records" in file.stem:
            try:
                clipped_records_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")

    if not sta_mag_dfs:
        raise custom_errors.NoStationsError(
            "No station magnitude data was found, please check the origin of the earthquake"
        )

    sta_mag_df = pd.concat(sta_mag_dfs, ignore_index=True)
    skipped_records_df = (
        pd.concat(skipped_records_dfs, ignore_index=True)
        if skipped_records_dfs
        else pd.DataFrame()
    )
    clipped_records_df = (
        pd.concat(clipped_records_dfs, ignore_index=True)
        if clipped_records_dfs
        else pd.DataFrame()
    )

    # Save the dataframes
    sta_mag_df.to_csv(
        flatfile_dir
        / file_structure.PreFlatfileNames.STATION_MAGNITUDE_TABLE_EXTRACTION,
        index=False,
    )
    skipped_records_df.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.EXTRACTION_SKIPPED_RECORDS,
        index=False,
    )
    clipped_records_df.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.CLIPPED_RECORDS,
        index=False,
    )
