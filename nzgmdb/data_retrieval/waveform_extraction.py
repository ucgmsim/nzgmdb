"""
Holds all the functions for extracting waveforms for the NZGMDB database from the FDSN Client.
"""

import functools
import http.client
import itertools
import multiprocessing as mp
import time
import warnings
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import Stream, UTCDateTime
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


def perform_ai_selection(st: Stream, ptime_est: UTCDateTime, ds_mean: float, ds_std: float):
    """
    Calculates the Arias Intensity for each trace for every channel over the time period of
    ptime_est to ptime_est + ds595 + 1std and selects the best 3 traces based on the highest
    contribution to the total Arias Intensity.

    Parameters
    ----------
    st : Stream
        The stream object containing the waveform data for every channel and location.
    ptime_est : UTCDateTime
        The estimated P-wave arrival time.
    ds_mean : float
        The mean of the log-transformed significant duration (ds595).
    ds_std : float
        The standard deviation of the log-transformed significant duration (ds595).

    Returns
    -------
    Stream
        A cleaned stream object containing only the best 3 traces based on Arias Intensity.
    """
    # Calculate Arias Intensity for each trace
    ai_values = {}
    for tr in st:
        ai = filtering.calculate_arias_intensity(tr, ptime_est)
        ai_values[tr.id] = ai

    # Sort traces by Arias Intensity in descending order
    sorted_traces = sorted(ai_values.items(), key=lambda item: item[1], reverse=True)

    # Select the top 3 traces
    selected_trace_ids = [trace_id for trace_id, _ in sorted_traces[:3]]
    selected_stream = st.select(id=",".join(selected_trace_ids))

    return selected_stream


def select_horizontal_pair(values: list[str]):
    """
    Selects the best horizontal channel pair from a list of available channels based on a predefined priority.

    Parameters
    ----------
    values : list of str
        A list of channel identifiers (e.g., ["1", "2", "X", "Y", "N", "E"]).

    Returns
    -------
    tuple or None
        A tuple containing the selected horizontal channel pair (e.g., ("1", "2")) or None if no valid pair is found.
    list of str
        A list of issues encountered during the selection process, such as incomplete pairs.
    """
    # Define valid pairs in priority order
    pair_priority = [
        ("1", "2"),  # highest priority
        ("X", "Y"),
        ("N", "E"),  # lowest priority
    ]
    found_pairs = []
    issues = []
    # Check each valid pair
    for a, b in pair_priority:
        if a in values and b in values:
            found_pairs.append((a, b))
    # Check for incomplete pairs (dangling values)
    for a, b in pair_priority:
        if (a in values) ^ (b in values):  # XOR = only one present
            issues.append(f"Incomplete pair: missing {b if a in values else a}")
    # Select based on priority
    if found_pairs:
        selected = found_pairs[0]
    else:
        selected = None
    return selected, issues


def check_trace_issues(st: Stream, record_id: str, station_extraction_row: pd.Series):
    """
    Checks for issues in the stream and manages them.
    1. Check if all the sample rates are the same, if not resample with interpolation and raise issue
    2. Check the final length of the traces, if greater than 3 then send to multi_trace_management
    3. Ensure traces all have the same length, if not trim to the shortest length

    Parameters
    ----------
    st : Stream
        The stream object containing the waveform data for every channel and location
    record_id : str
        The record id for the stream, used for raising issues and skipped records
    station_extraction_row : pd.Series
        A row from the station extraction table containing the parameters for waveform extraction.

    Returns
    -------
    Stream
        A cleaned stream object with issues managed or None if the issues could not be managed.
    pd.DataFrame
        A DataFrame containing the skipped record if the issues could not be managed or None if the record wasn't skipped.
    pd.DataFrame
        A DataFrame containing any issues raised during the management of the stream that need to be flagged.
    """
    raised_issues = []

    # Check if there is less than 3 traces
    if len(st) < 3:
        skipped_record = pd.DataFrame(
            {"record_id": [record_id], "reason": ["Less than 3 traces"]}
        )
        return None, skipped_record, raised_issues

    # Check if all the sample rates are the same
    samples = [tr.stats.sampling_rate for tr in st]
    if len(set(samples)) > 1:
        # If they are different take the highest and resample with the others using interpolation
        st = st.select(sampling_rate=max(samples))
        st.merge(fill_value="interpolate")
        raised_issues.append(
            pd.DataFrame(
                {"record_id": [record_id], "reason": ["Different sample rates"]}
            )
        )

    # Check if there is any data between starting noise -> ds595 * 1std
    config = cfg.Config()
    ptime_est = UTCDateTime(station_extraction_row["ptime_est"])
    ds_mean = station_extraction_row["ds_mean"]
    ds_std = station_extraction_row["ds_std"]
    ds_end_time = np.exp(ds_mean) * np.exp(ds_std)
    pre_event_time_difference = config.get_value("pre_event_time_difference")
    investigation_start_time = ptime_est - pre_event_time_difference
    investigation_end_time = ptime_est + ds_end_time

    # Make a copy of the traces to check for data in the investigation window
    st_check = st.copy()
    st_check.trim(investigation_start_time, investigation_end_time)
    if len(st_check) == 0 or all([tr.stats.npts == 0 for tr in st_check]):
        skipped_record = pd.DataFrame(
            {
                "record_id": [record_id],
                "reason": ["No data from noise to ds595 + 1std"],
            }
        )
        return None, skipped_record, raised_issues

    # Perform a check to see if there is any offset in the traces missing data between noise and ds595 + 1std
    # Group traces by channel into separate arrays
    channel_traces = {}
    for trace in st_check:
        channel = trace.stats.channel
        channel_traces.setdefault(channel, []).append(trace)

    # Sort traces by start time for each channel
    for channel, traces in channel_traces.items():
        channel_traces[channel] = sorted(
            traces, key=lambda tr: tr.stats.starttime.timestamp
        )

    # Find the maximum number of traces among all channels
    max_len = max(len(traces) for traces in channel_traces.values())
    multi_offset = False
    total_duration = investigation_end_time - investigation_start_time
    percentage_gap_allowed = config.get_value("percentage_gap_allowed")

    # Check for offsets in the traces
    for idx in range(max_len):
        traces_at_idx = []
        for channel, traces in channel_traces.items():
            if idx < len(traces):
                traces_at_idx.append(traces[idx])
        # Compare all pairs of traces at this index
        for t1, t2 in itertools.combinations(traces_at_idx, 2):
            start_diff = abs(
                t1.stats.starttime.timestamp - t2.stats.starttime.timestamp
            )
            end_diff = abs(t1.stats.endtime.timestamp - t2.stats.endtime.timestamp)
            if (
                start_diff / total_duration > percentage_gap_allowed
                or end_diff / total_duration > percentage_gap_allowed
            ):
                multi_offset = True
                break
        if multi_offset:
            break

    if multi_offset:
        skipped_record = pd.DataFrame(
            {
                "record_id": [record_id],
                "reason": [
                    "Offset in traces missing data between noise and ds595 + 1std"
                ],
            }
        )
        return None, skipped_record, raised_issues

    # Check if multi-trace issue (greater than 3 traces)
    if len(st) > 3:
        # Check if there is multiple Horizontal channel pairs
        h_channels = {
            trace.stats.channel[-1] for trace in st if trace.stats.channel[-1] != "Z"
        }
        if len(h_channels) > 2:
            # Here we need to select the correct horizontal channels
            pair, issues = select_horizontal_pair(list(h_channels))
            if pair:
                st = (
                    st.select(channel=f"*{pair[0]}")
                    + st.select(channel=f"*{pair[1]}")
                    + st.select(channel="*Z")
                )
                raised_issues.append(
                    pd.DataFrame(
                        {
                            "record_id": [record_id],
                            "reason": [
                                f"Multiple horizontal channels, selected pair: {pair[0]}, {pair[1]}"
                            ],
                        }
                    )
                )
                for issue in issues:
                    raised_issues.append(
                        pd.DataFrame(
                            {
                                "record_id": [record_id],
                                "reason": [issue],
                            }
                        )
                    )
            else:
                skipped_record = pd.DataFrame(
                    {
                        "record_id": [record_id],
                        "reason": [
                            "Multiple horizontal channels with no valid pair (1/2, X/Y, N/E)"
                        ],
                    }
                )
                return None, skipped_record, raised_issues

        # Check for overlapping data if still greater than 3 traces
        if len(st) > 3:
            # Group traces by channel into separate arrays
            channel_traces = {}
            for trace in st:
                channel = trace.stats.channel
                channel_traces.setdefault(channel, []).append(trace)

            # Sort traces by start time for each channel
            for channel, traces in channel_traces.items():
                channel_traces[channel] = sorted(
                    traces, key=lambda tr: tr.stats.starttime.timestamp
                )

            overlapping = False
            is_large = False
            different_data = False
            is_large_overlap = config.get_value("is_large_overlap")

            # Add checks that there is overlapping data
            for channel, traces in channel_traces.items():
                for t1, t2 in itertools.combinations(traces, 2):
                    latest_start = max(t1.stats.starttime, t2.stats.starttime)
                    earliest_end = min(t1.stats.endtime, t2.stats.endtime)
                    if latest_start < earliest_end:
                        overlapping = True
                        # we have overlapping, just need to confirm the type
                        overlap_duration = earliest_end - latest_start
                        if overlap_duration / total_duration > is_large_overlap:
                            is_large = True
                            # Check if the data is different in the large overlapping region
                            overlap_start_idx1 = int(
                                (latest_start - t1.stats.starttime)
                                * t1.stats.sampling_rate
                            )
                            overlap_start_idx2 = int(
                                (latest_start - t2.stats.starttime)
                                * t2.stats.sampling_rate
                            )
                            overlap_end_idx1 = int(
                                (earliest_end - t1.stats.starttime)
                                * t1.stats.sampling_rate
                            )
                            overlap_end_idx2 = int(
                                (earliest_end - t2.stats.starttime)
                                * t2.stats.sampling_rate
                            )
                            data1 = t1.data[overlap_start_idx1:overlap_end_idx1]
                            data2 = t2.data[overlap_start_idx2:overlap_end_idx2]
                            # Compare data arrays
                            if not (
                                data1.shape == data2.shape
                                and np.isclose(data1, data2).all()
                            ):
                                different_data = True

            if overlapping:
                raised_issues.append(
                    pd.DataFrame(
                        {
                            "record_id": [record_id],
                            "reason": [
                                (
                                    "Large overlapping data between traces"
                                    if is_large
                                    else "Small overlapping data between traces"
                                )
                            ],
                        }
                    )
                )
                if different_data:
                    skipped_record = pd.DataFrame(
                        {
                            "record_id": [record_id],
                            "reason": [
                                "Large overlapping data with different data between traces"
                            ],
                        }
                    )
                    return None, skipped_record, raised_issues
                else:
                    # Keep the longest trace for each channel
                    longest_traces = []
                    for channel, traces in channel_traces.items():
                        longest_trace = max(
                            traces, key=lambda tr: tr.stats.endtime - tr.stats.starttime
                        )
                        longest_traces.append(longest_trace)
                    st = Stream(traces=longest_traces)
                    raised_issues.append(
                        pd.DataFrame(
                            {
                                "record_id": [record_id],
                                "reason": [
                                    "Kept longest trace for each channel due to overlapping duplicate data"
                                ],
                            }
                        )
                    )

    if len(st) > 3:
        # This is now a basic multi-trace problem with no extreme issues
        # Utilise Arias-Intensity to select the best 3 traces
        st_preffered =

        # If still greater than 3 traces then skip due to an edge case we haven't managed
        skipped_record = pd.DataFrame(
            {
                "record_id": [record_id],
                "reason": [
                    "More than 3 traces after managing multi-trace issues, edge case not managed"
                ],
            }
        )
        return None, skipped_record, raised_issues

    # Ensure traces all have the same length
    starttime_trim = max([tr.stats.starttime for tr in st])
    endtime_trim = min([tr.stats.endtime for tr in st])
    # Check that the start time is before the end time
    if starttime_trim > endtime_trim:
        skipped_record = pd.DataFrame(
            {
                "record_id": [record_id],
                "reason": [
                    "Start time after end time when trimming to common length"
                ],
            }
        )
        return None, skipped_record, raised_issues
    st.trim(starttime_trim, endtime_trim)

    return st, None, raised_issues


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
    ptime_est = UTCDateTime(station_extraction_row["ptime_est"])
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
    list
        A list of DataFrames containing any multi-trace issues raised during the extraction.
    """
    sta_mag_line, skipped_records, clipped_records, multi_trace_issues = [], [], [], []
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
        event_only_record_ids = only_record_ids[
            only_record_ids["record_id"].str.startswith(f"{event_id}_")
        ]
        site_only_record_ids = event_only_record_ids[
            event_only_record_ids["record_id"].str.contains(f"_{station}_")
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
    mseeds = []
    for chan, loc in unique_channels:
        # Each unique channel and location pair is a new mseed file
        st_new = st.select(location=loc, channel=f"{chan}?")
        record_id = f"{event_id}_{st_new[0].stats.station}_{st_new[0].stats.channel[:2]}_{st_new[0].stats.location}"

        # Check trace issues
        st_revised, skipped, issues = check_trace_issues(
            st_new, record_id, station_extraction_row
        )

        multi_trace_issues.extend(issues)
        # Add to the skipped records if any were raised
        if skipped:
            skipped_records.append(skipped)
            continue
        else:
            mseeds.append(st_revised)


    # Get the station magnitudes
    station_magnitudes = [
        mag
        for mag in event_cat.station_magnitudes
        if mag.waveform_id.station_code == station
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
        # clip = filtering.get_clip_probability(mag, r_hyp, mseed)

        # threshold = config.get_value("clip_threshold")

        # Check if the record should be dropped
        # if clip > threshold:
        #     stats = mseed[0].stats
        #     clipped_records.append(
        #         [
        #             f"{event_id}_{stats.station}_{stats.channel[:2]}_{stats.location}",
        #             "Clipped",
        #         ]
        #     )

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

    return sta_mag_line, skipped_records, clipped_records, multi_trace_issues


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

    # If only_record_ids is provided, filter down the station_extraction_table
    if only_record_ids is not None:
        # Make a evid-sta column in the station_extraction_table
        station_extraction_table["evid_sta"] = (
            station_extraction_table["evid"] + "_" + station_extraction_table["sta"]
        )
        # Mkae a evid_sta column in the only_record_ids
        only_record_ids["evid_sta"] = (
            only_record_ids["record_id"].str.split("_").str[0]
            + "_"
            + only_record_ids["record_id"].str.split("_").str[1]
        )
        # Filter the station_extraction_table to only include the evid_sta in the only_record_ids
        station_extraction_table = station_extraction_table[
            station_extraction_table["evid_sta"].isin(only_record_ids["evid_sta"])
        ].copy()

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
            print(f"Processing batch {batch_index + 1}/{len(index_batches)}")
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
            sta_mag_data, skipped_records, clipped_records, multi_trace_issues = [], [], [], []
            for result in results:
                (
                    finished_sta_mag_data,
                    finished_skipped_records,
                    finished_clipped_records,
                    finished_multi_trace_issues,
                ) = result
                sta_mag_data.extend(finished_sta_mag_data)
                skipped_records.extend(finished_skipped_records)
                clipped_records.extend(finished_clipped_records)
                multi_trace_issues.extend(finished_multi_trace_issues)

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

            if len(multi_trace_issues) > 0:
                multi_trace_issues_df = pd.concat(multi_trace_issues, ignore_index=True)
                multi_trace_issues_df.to_csv(
                    batch_dir / f"multi_trace_issues_{batch_index}.csv", index=False
                )

    # Combine all the event and sta_mag dataframes
    sta_mag_dfs = []
    skipped_records_dfs = []
    clipped_records_dfs = []
    multi_trace_issues_dfs = []

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
        elif "multi_trace_issues" in file.stem:
            try:
                multi_trace_issues_dfs.append(pd.read_csv(file))
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
    multi_trace_issues_df = (
        pd.concat(multi_trace_issues_dfs, ignore_index=True)
        if multi_trace_issues_dfs
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
    multi_trace_issues_df.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.MULTI_TRACE_ISSUE_RECORDS,
        index=False,
    )
