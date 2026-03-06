"""
Holds all the functions for extracting waveforms for the NZGMDB database from the FDSN Client.
"""

import functools
import http.client
import itertools
import multiprocessing as mp
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy as sp
from obspy import Stream, Trace, UTCDateTime, read_inventory, read
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import (
    FDSNNoDataException,
    FDSNServiceUnavailableException,
    FDSNTooManyRequestsException,
)
from obspy.io.mseed import InternalMSEEDError, ObsPyMSEEDFilesizeTooSmallError
from pandas.errors import EmptyDataError

from nzgmdb.data_processing import filtering, multi_event
from nzgmdb.data_retrieval import inventory_xml
from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors, file_structure
from nzgmdb.mseed_management import creation


class StationExtractionResult(NamedTuple):
    """
    Container for waveform extraction results from a single station.
    """

    sta_mag_line: list[list[object]]
    skipped_records: list[object]
    clipped_records: list[list[object]]
    multi_trace_issues: list[pd.DataFrame]
    multi_event_records: list[list[object]]


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
                )
            break
        except FDSNTooManyRequestsException:
            print(f"Error getting waveforms for {net}.{sta}")
            print("Too many requests - HTTP Status code: 429")
            print("Retrying in 120 seconds...")
            time.sleep(120)  # Wait for 2 minutes before retrying
            # reset attempt count
            attempt = 0
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


def get_arias_intensity_norm(
    trace: Trace,
):
    """
    Calculate the Arias intensity from a trace object.

    Parameters
    ----------
    trace : Trace
        The trace object containing the waveform data

    Returns
    -------
    np.ndarray
        The Arias intensity as a 2D array with time and normalized intensity values
    """
    g = 9.81
    dt = trace.stats.delta
    a_sq = trace.data**2.0

    arias_intensity = (
        np.pi / (2 * g) * sp.integrate.cumulative_trapezoid(a_sq[:-1], dx=dt, initial=0)
    )

    return arias_intensity


def perform_ai_selection(
    st: Stream, ptime_est: UTCDateTime, ds_mean: float, ds_std: float
):
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
    int
        The index of the selected trace in the original stream. None if there was an issue.
    bool
        A boolean indicating if there was an issue with the selection process. (Groups did not agree)
    """
    # Copy the stream
    st_copy = st.copy()
    # Detrend the mseed
    st_copy.detrend("demean")
    st_copy.detrend("linear")
    # Filter the mseed (If using a lower frequency sensor e.g. BN then you will get a warning, however it will still filter and makes no difference so we ignore the warning)
    warnings.filterwarnings(
        "ignore",
        message="Selected high corner frequency*",
        category=UserWarning,
        module="obspy.signal.filter",
    )
    st_copy.filter("bandpass", freqmin=0.1, freqmax=40)

    # Get the unique ending channel ids
    unique_channel_endings = list(set([tr.stats.channel[-1] for tr in st_copy]))
    trace_indexs = []
    traces = []
    loc = st_copy[0].stats.location
    chan = st_copy[0].stats.channel[:2]

    for unique_channel_ending in unique_channel_endings:
        # We need to group together each component with the same ending channel str e.g. HN1 the 1
        group = st_copy.select(
            location=loc,
            channel=f"{chan}{unique_channel_ending}",
        )
        # For each trace compute the AI and keep the highest
        for i, tr in enumerate(group):
            # Trim the trace to ptime_est to ptime_est + ds595 + 1std
            ds_end_time = np.exp(ds_mean) * np.exp(ds_std)
            tr_copy = tr.copy()
            tr_copy.trim(ptime_est, ptime_est + ds_end_time, pad=True, fill_value=0)
            # Get the Arias Intensity
            arias_intensity = get_arias_intensity_norm(tr_copy)
            if i == 0:
                arias_intensity_max = arias_intensity[-1]
                arias_intensity_max_index = 0
            elif arias_intensity[-1] > arias_intensity_max:
                arias_intensity_max = arias_intensity[-1]
                arias_intensity_max_index = i
        trace_indexs.append(arias_intensity_max_index)
        # Grab the original trace from the st and group to get the correct trace
        group_original = st.select(
            location=loc,
            channel=f"{chan}{unique_channel_ending}",
        )
        traces.append(group_original[arias_intensity_max_index])
    # Check all the groups agree
    if len(set(trace_indexs)) == 1:
        # We combine the traces into a new stream
        st = Stream(traces=traces)
        trace_index = trace_indexs[0]
        issue = False
    else:
        # Raise an issue here as the groups do not agree
        issue = True
        trace_index = None

    return st, trace_index, issue


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
    pair_priority = [("1", "2"), ("X", "Y"), ("N", "E")]
    issues = []
    selected = None
    for a, b in pair_priority:
        has_a, has_b = a in values, b in values
        if selected is None and has_a and has_b:
            selected = (a, b)
        elif has_a ^ has_b:
            issues.append(f"Incomplete pair: missing {b if has_a else a}")
    return selected, issues


def check_trace_issues(st: Stream, record_id: str, station_extraction_row: pd.Series):
    """
    Checks for issues in the stream and manages them.
    1. Less than 3 traces -> Skip
    2. Different sample rates -> Resample to highest sample rate
    3. No data between starting noise -> ds595 * 1std -> Skip
    4. Offset in traces missing data between noise and ds595 * 1std -> Skip
    5. Multi-trace issue (still greater than 3 traces)
        a. Multiple Horizontal channel pairs -> Select a pair or Skip
        b. Overlapping data with different data -> Skip
        c. Overlapping data with same data -> Keep longest trace for each channel
        d. Small gaps (20 data points) -> Raise issue but keep
        e. Changes in timesteps -> Raise issue but keep
        f. Basic multi-trace problem -> Use Arias Intensity to select best 3 traces or Skip

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
    main_intensity_check = st.copy()
    main_intensity_check.trim(ptime_est, investigation_end_time)
    if len(main_intensity_check) == 0 or all(
        [tr.stats.npts == 0 for tr in main_intensity_check]
    ):
        # This means we have an emtpy stream
        skipped_record = pd.DataFrame(
            {
                "record_id": [record_id],
                "reason": ["No data from ptime_est to ds595 + 1std"],
            }
        )
        return None, skipped_record, raised_issues

    st_check = st.copy()
    st_check.trim(investigation_start_time, investigation_end_time)
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
        found_small_gap = False
        # Add extra checks for small trace gaps (20 data points)
        for channel, traces in channel_traces.items():
            if len(traces) > 1:
                for i in range(len(traces) - 1):
                    gap = traces[i + 1].stats.starttime - traces[i].stats.endtime
                    if gap <= 20 * (1 / traces[i].stats.sampling_rate):
                        # We have a small gap
                        # Categorise the gaps into noise, main_intensity, tail
                        gap_start = traces[i].stats.endtime
                        gap_end = traces[i + 1].stats.starttime
                        if gap_start < ptime_est:
                            category = "noise"
                        elif (
                            gap_start >= ptime_est and gap_end <= investigation_end_time
                        ):
                            category = "main_intensity"
                        else:
                            category = "tail"
                        raised_issues.append(
                            pd.DataFrame(
                                {
                                    "record_id": [record_id],
                                    "reason": [
                                        f"Found small gap during {category} section of waveform"
                                    ],
                                }
                            )
                        )
                        found_small_gap = True
                        break  # Only need to find one small gap to raise the issue
            if found_small_gap:
                break

        # Add check for changes in the timesteps
        # Also categorise the gaps into noise, main_intensity, tail
        found_timestep_change = False
        if len(st) > 3:
            for channel, traces in channel_traces.items():
                if len(traces) > 1:
                    for i in range(len(traces) - 1):
                        first_tr_end = traces[i].stats.endtime
                        second_tr_start = traces[i + 1].stats.starttime
                        # Check if the traces are still aligned with delta
                        difference_time = second_tr_start - first_tr_end
                        # Check if the difference_time is divisible by the delta
                        delta_diff = difference_time % traces[i].stats.delta
                        if not (
                            np.isclose(delta_diff, 0)
                            or np.isclose(delta_diff, traces[i].stats.delta)
                        ):
                            # We have a change in timestep between traces
                            if second_tr_start < ptime_est:
                                category = "noise"
                            elif (
                                second_tr_start >= ptime_est
                                and second_tr_start <= investigation_end_time
                            ):
                                category = "main_intensity"
                            else:
                                category = "tail"
                            raised_issues.append(
                                pd.DataFrame(
                                    {
                                        "record_id": [record_id],
                                        "reason": [
                                            f"Found change in timestep during {category} section of waveform"
                                        ],
                                    }
                                )
                            )
                            found_timestep_change = True
                            break
                if found_timestep_change:
                    break

        # This is now a basic multi-trace problem with no extreme issues
        # Utilise Arias-Intensity to select the best 3 traces
        st_preffered, index_selected, issue = perform_ai_selection(
            st, ptime_est, ds_mean, ds_std
        )

        if issue:
            skipped_record = pd.DataFrame(
                {
                    "record_id": [record_id],
                    "reason": [
                        "Could not agree on best traces using Arias Intensity selection"
                    ],
                }
            )
            return None, skipped_record, raised_issues

        if len(st_preffered) > 3:
            skipped_record = pd.DataFrame(
                {
                    "record_id": [record_id],
                    "reason": [
                        "More than 3 traces after Arias Intensity selection, edge case not managed"
                    ],
                }
            )
            return None, skipped_record, raised_issues
        else:
            st = st_preffered
            raised_issues.append(
                pd.DataFrame(
                    {
                        "record_id": [record_id],
                        "reason": [
                            "Reduced to 3 traces using Arias Intensity selection"
                        ],
                    }
                )
            )
            raised_issues.append(
                pd.DataFrame(
                    {
                        "record_id": [record_id],
                        "reason": [f"Selected trace: {index_selected+1}"],
                    }
                )
            )

    # Ensure traces all have the same length
    starttime_trim = max([tr.stats.starttime for tr in st])
    endtime_trim = min([tr.stats.endtime for tr in st])
    # Check that the start time is before the end time
    if starttime_trim > endtime_trim:
        skipped_record = pd.DataFrame(
            {
                "record_id": [record_id],
                "reason": ["Start time after end time when trimming to common length"],
            }
        )
        return None, skipped_record, raised_issues
    st.trim(starttime_trim, endtime_trim)

    return st, None, raised_issues


def get_station_window(
    station_extraction_row: pd.Series,
):
    """
    Get the start and end time for the waveform extraction window for a station based on the parameters in the station extraction table.

    Parameters
    ----------
    station_extraction_row : pd.Series
        A row from the station extraction table containing the parameters for waveform extraction.

    Returns
    -------
    tuple of UTCDateTime
        A tuple containing the start time and end time for the waveform extraction window.
    """
    # Extract the parameters from the row
    r_hyp = station_extraction_row["r_hyp"]
    ptime_est = UTCDateTime(station_extraction_row["ptime_est"])
    ds_mean = station_extraction_row["ds_mean"]
    ds_std = station_extraction_row["ds_std"]

    # Get the config values
    config = cfg.Config()
    pre_event_time_difference = config.get_value("pre_event_time_difference")

    # Compute the ds multiplier time
    # NOTE: This is based on an equation derived from statistical analysis of NZGMDB data
    # by Aaron, when looking at impacts of ds_std multiplier on picking up multi-event records.
    ds_std_multiplier = 0.8 / (1 + np.exp(-0.035 * (r_hyp - 140))) + 2.2

    start_time = ptime_est - pre_event_time_difference
    # Note: both ds_mean and ds_std are in logspace
    end_time = ptime_est + np.exp(ds_mean) * np.exp(ds_std_multiplier * ds_std)

    return start_time, end_time


def get_tmp_array_stream(
    tmp_array_dir: Path,
    net: str,
    sta: str,
    start_time: UTCDateTime,
    end_time: UTCDateTime,
):
    """
    Get the initial stream of waveforms for a station from the temporary array storage location.

    Parameters
    ----------
    tmp_array_dir : Path
        The directory where the temporary array waveform files are stored.
    net : str
        The network code to retrieve waveforms for.
    sta : str
        The station code to retrieve waveforms for.
    start_time : UTCDateTime
        The start time of the waveform data to retrieve.
    end_time : UTCDateTime
        The end time of the waveform data to retrieve.

    Returns
    -------
    Stream
        An ObsPy Stream object containing the waveform data for the specified station.
    """
    net_dir = tmp_array_dir / net
    if not net_dir.is_dir():
        return None

    st = Stream()

    pattern = f"{net}_{sta}_*"
    selected_files = []
    for sta_dir in sorted(p for p in net_dir.glob(pattern) if p.is_dir()):

        for f in sta_dir.glob("*.mseed"):
            # Example filename
            # Y3.CASS..HHN__20090312T000000Z__20090411T000000Z.mseed
            parts = f.name.split("__")

            file_start = UTCDateTime(parts[1])
            file_end = UTCDateTime(parts[2].replace(".mseed", ""))

            # Check overlap
            if file_end >= start_time and file_start <= end_time:
                selected_files.append(f)

    if not selected_files:
        return None

    # Read files
    for f in sorted(selected_files):
        st += read(str(f))

    # Merge overlapping / adjacent segments
    st.merge(method=1, fill_value=None)

    # Trim to exact window
    st.trim(start_time, end_time)

    return st


def extract_station_info(
    station_extraction_row: pd.Series,
    main_dir: Path,
    event_catalogues: dict,
    extraction_table: pd.DataFrame,
    only_record_ids: pd.DataFrame = None,
    tmp_array_dir: Path = None,
) -> StationExtractionResult:
    """
    Extract the waveform data for a single station based on the extraction parameters.

    Parameters
    ----------
    station_extraction_row : pd.Series
        A row from the station extraction table containing the parameters for waveform extraction.
    main_dir : Path
        The main directory of the NZGMDB results (Highest Level Directory).
    event_catalogues : dict
        A dictionary of event catalogues indexed by event ID.
    extraction_table : pd.DataFrame
        The full extraction table containing all extraction parameters.
    only_record_ids : pd.DataFrame, optional
        A DataFrame containing a subset of record IDs to use for extraction, if provided.
    tmp_array_dir : Path, optional
        The directory where the temporary array waveform files are stored, if using temporary array storage for waveforms.

    Returns
    -------
    StationExtractionResult
        Named result object containing station magnitude data, skipped records, clipped records,
        multi-trace issues, and multi-event record scores.
    """
    (
        sta_mag_line,
        skipped_records,
        clipped_records,
        multi_trace_issues,
        multi_event_records,
    ) = ([], [], [], [], [])
    # Extract the parameters from the row
    provider = station_extraction_row["provider"]
    event_id = station_extraction_row["evid"]
    station = station_extraction_row["sta"]
    network = station_extraction_row["net"]
    event_mag = station_extraction_row["mag"]
    pref_mag_type = station_extraction_row["pref_mag_type"]
    r_hyp = station_extraction_row["r_hyp"]

    # Filter down the extraction table to the same station and other events
    sync_check_extraction_table = extraction_table[
        (extraction_table["sta"] == station) & (extraction_table["evid"] != event_id)
    ]

    # Get the catalogue information
    event_cat = event_catalogues[event_id]

    # Obtain the station channel codes and location
    config = cfg.Config()
    channel_codes = config.get_value("channel_codes")
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

    start_time, end_time = get_station_window(station_extraction_row)
    net = station_extraction_row["net"]
    sta = station_extraction_row["sta"]

    if provider == "GEONET":
        # Get the Stream
        client = FDSN_Client("GEONET")
        st = get_inital_stream(
            start_time, end_time, channel_codes, location, client, net, sta
        )
    else:
        # Get the stream from the tmp array storage location
        st = get_tmp_array_stream(tmp_array_dir, net, sta, start_time, end_time)

    # Check that data was found
    if st is None:
        skipped_records.append(
            pd.DataFrame(
                {"record_id": [f"{event_id}_{station}"], "reason": ["No Waveform Data"]}
            )
        )
        return StationExtractionResult(
            sta_mag_line=sta_mag_line,
            skipped_records=skipped_records,
            clipped_records=clipped_records,
            multi_trace_issues=multi_trace_issues,
            multi_event_records=multi_event_records,
        )

    # Get the inventory xml file
    xml_dir = file_structure.get_stationxml_dir(main_dir)
    # Load the inventory information
    inventory_file = xml_dir / f"{station}.xml"
    inventory = read_inventory(inventory_file) if inventory_file.is_file() else None

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
        if skipped is not None:
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
                    pd.DataFrame(
                        {
                            "record_id": [
                                f"{event_id}_{stats.station}_{stats.channel}_{stats.location}"
                            ],
                            "reason": ["All 0's"],
                        }
                    )
                )
                continue
        except TypeError:
            stats = mseed[0].stats
            skipped_records.append(
                pd.DataFrame(
                    {
                        "record_id": [
                            f"{event_id}_{stats.station}_{stats.channel}_{stats.location}"
                        ],
                        "reason": ["TypeError when checking for all 0's"],
                    }
                )
            )

        # Calculate clip to determine if the record should be dropped
        clip = filtering.get_clip_probability(event_mag, r_hyp, mseed)

        threshold = config.get_value("clip_threshold")
        stats = mseed[0].stats
        record_id = f"{event_id}_{stats.station}_{stats.channel[:2]}_{stats.location}"

        # Check if the record should be dropped
        if clip > threshold:

            clipped_records.append(
                [
                    record_id,
                    "Clipped",
                ]
            )

        # Check for jerks
        has_jerk = filtering.get_jerk(mseed)
        if has_jerk:
            clipped_records.append(
                [
                    record_id,
                    "Jerk",
                ]
            )

        # Check for multi-event flagging
        start_time, end_time, stalta_score, sync_event = (
            multi_event.compute_multi_event_scores(
                mseed.copy(), sync_check_extraction_table, inventory=inventory
            )
        )
        # Add to the multi_event_records list
        multi_event_records.append(
            [
                record_id,
                start_time.isoformat(),
                end_time.isoformat(),
                stalta_score,
                sync_event,
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

    return StationExtractionResult(
        sta_mag_line=sta_mag_line,
        skipped_records=skipped_records,
        clipped_records=clipped_records,
        multi_trace_issues=multi_trace_issues,
        multi_event_records=multi_event_records,
    )


def extract_waveforms(
    main_dir: Path,
    station_extraction_table_ffp: Path,
    n_procs: int = 1,
    only_record_ids_ffp: Path = None,
    batch_size: int = 1000,
    tmp_array_dir: Path = None,
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
    tmp_array_dir : Path, optional
        The directory where the temporary array waveform files are stored, if using temporary array storage for waveforms.
    """
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
        # Make a evid_sta column in the only_record_ids
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
    client = FDSN_Client("GEONET")

    for batch_index, batch_indices in enumerate(index_batches):
        if batch_index not in processed_suffixes:
            print(f"Processing batch {batch_index + 1}/{len(index_batches)}")
            batch_rows = station_extraction_table.loc[batch_indices]

            # Get the catalogue information
            fetched_catalog = False
            attempts = 0
            while not fetched_catalog:
                attempts += 1
                if attempts > 5:
                    raise Exception(
                        f"Failed to fetch event catalog after {attempts} attempts."
                    )
                try:
                    catalog_dict = {
                        event_id: client.get_events(eventid=event_id)[0]
                        for event_id in batch_rows["evid"].unique()
                    }
                    fetched_catalog = True
                except FDSNTooManyRequestsException:
                    print(f"Error getting catalog for batch {batch_index}")
                    print("Too many requests - HTTP Status code: 429")
                    print("Retrying in 120 seconds...")
                    time.sleep(120)  # Wait for 2 minutes before retrying

            with mp.Pool(n_procs) as pool:
                results = pool.map(
                    functools.partial(
                        extract_station_info,
                        main_dir=main_dir,
                        event_catalogues=catalog_dict,
                        extraction_table=station_extraction_table,
                        only_record_ids=only_record_ids,
                        tmp_array_dir=tmp_array_dir,
                    ),
                    (row for _, row in batch_rows.iterrows()),
                )

            # Extract the results
            (
                sta_mag_data,
                skipped_records,
                clipped_records,
                multi_trace_issues,
                multi_event_records,
            ) = (
                [],
                [],
                [],
                [],
                [],
            )
            for result in results:
                (
                    finished_sta_mag_data,
                    finished_skipped_records,
                    finished_clipped_records,
                    finished_multi_trace_issues,
                    finished_multi_event_records,
                ) = result
                sta_mag_data.extend(finished_sta_mag_data)
                skipped_records.extend(finished_skipped_records)
                clipped_records.extend(finished_clipped_records)
                multi_trace_issues.extend(finished_multi_trace_issues)
                multi_event_records.extend(finished_multi_event_records)

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
                skipped_records_df = pd.concat(skipped_records)
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

            if len(multi_event_records) > 0:
                multi_event_records_df = pd.DataFrame(
                    multi_event_records,
                    columns=[
                        "record_id",
                        "start_time",
                        "end_time",
                        "stalta_score",
                        "sync_event",
                    ],
                )
                multi_event_records_df.to_csv(
                    batch_dir / f"multi_event_records_{batch_index}.csv", index=False
                )

    # Grab all the station xmls and write them as outputs
    unique_sites = station_extraction_table["sta"].unique()
    print(f"Fetching station XML metadata for {len(unique_sites)} unique sites")
    inventory_xml.fetch_and_save_inventory(main_dir, unique_sites)
    print("Station XML metadata fetching complete.")

    # Combine all the event and sta_mag dataframes
    sta_mag_dfs = []
    skipped_records_dfs = []
    clipped_records_dfs = []
    multi_trace_issues_dfs = []
    multi_event_records_dfs = []

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
        elif "multi_event_records" in file.stem:
            try:
                multi_event_records_dfs.append(pd.read_csv(file))
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
    multi_event_records_df = (
        pd.concat(multi_event_records_dfs, ignore_index=True)
        if multi_event_records_dfs
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
    multi_event_records_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.MULTI_EVENT_TABLE,
        index=False,
    )
