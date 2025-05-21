"""
This module contains functions for creating mseed files from the waveform data from the FDSN client
"""

import http
import http.client
import time
import warnings
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

import mseedlib
import numpy as np
import pandas as pd
from obspy import Trace, Stream, UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import (
    FDSNNoDataException,
    FDSNServiceUnavailableException,
)
from obspy.core.event import Origin
from obspy.geodetics import kilometers2degrees
from obspy.io.mseed import InternalMSEEDError, ObsPyMSEEDFilesizeTooSmallError
from obspy.taup import TauPyModel
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

from empirical.util import classdef, openquake_wrapper_vectorized, z_model_calculations
from nzgmdb.management import config as cfg


def get_arias_intensity_norm(
    trace: Trace,
):
    """
    Calculate the normalized Arias intensity from a trace object.

    Parameters
    ----------
    trace : Trace
        The trace object containing the waveform data

    Returns
    -------
    Ia_norm : np.ndarray
        The normalized Arias intensity as a 2D array with time and normalized intensity values
    """
    g = 9.81
    dt = trace.stats.delta
    npts = trace.stats.npts

    t = np.linspace(0, (npts - 1) * dt, npts)
    a = trace.data  # acceleration in m/s^2

    a_sq = a**2.0
    Ia_1col = np.zeros(npts)

    for i in range(1, npts):
        Ia_1col[i] = Ia_1col[i - 1] + np.pi / (2 * g) * a_sq[i - 1] * dt

    Ia_peak = float(Ia_1col[-1])
    Ia_norm_1col = Ia_1col / Ia_peak
    Ia_norm = np.column_stack((t, Ia_norm_1col))

    return Ia_1col, Ia_norm


class WaveformState(StrEnum):
    FIND_WAVEFORM = "find_waveform"
    FIND_WAVEFORM_END = "find_waveform_end"
    FIND_NOISE_SECTION = "find_noise_section"


def get_sections(ds_npts: int, Ia_norm_diff: np.ndarray, Ia_norm: np.ndarray):
    # Stores each of the split sections and where they should be cut
    sections = [0]
    # Counter to keep track of the number of points in the waveform and when it goes above ds_npts
    counter = 0
    cur_i = 0

    # We only need to check 15% Ds for indictaing the start of a waveform
    # ds_npts_find_waveform = max(ds_npts, 4)
    ds_npts_find_waveform = ds_npts

    waveform_state = WaveformState.FIND_WAVEFORM
    # Travel from the start to the end of the IA_norm_diff
    for i, Ia_diff_value in enumerate(Ia_norm_diff):
        if waveform_state == WaveformState.FIND_WAVEFORM:
            if Ia_diff_value > 0.01:
                # If the value is greater than 0.01, check if the counter is greater than ds_npts_find_waveform
                # Or if the Ia_norm value is greater than 0.05 then we have found a waveform
                if counter >= ds_npts_find_waveform or Ia_diff_value > 0.05:
                    # Confirmed a waveform section, try find the end of the waveform
                    waveform_state = WaveformState.FIND_WAVEFORM_END
                    counter = 0
                    cur_i = i
                else:
                    # Increment the counter
                    counter += 1
            else:
                # Reset the counter
                counter = 0
        elif waveform_state == WaveformState.FIND_WAVEFORM_END:
            # Check if the value is less than 0.01
            if Ia_diff_value < 0.01:
                # Increment the counter
                counter += 1
            else:
                # Reset the counter
                counter = 0
                cur_i = i
            if counter >= ds_npts:
                # Add the index to the sections (Of the last value higher than noise)
                sections.append(cur_i)
                counter = 0
                waveform_state = WaveformState.FIND_NOISE_SECTION
        else:
            # We are in the noise section, check if the value is greater than 0.01
            # And only count as a section to end on if the Ia_norm value is less than 0.85
            # Meaning we have another potential waveform to extract
            if Ia_diff_value > 0.01 and (Ia_norm[i, 1] < 0.85 or Ia_diff_value > 0.05):
                # If it is, we have found the end of noise and can look for a waveform
                waveform_state = WaveformState.FIND_WAVEFORM
                counter = 1  # This is the first point of the potential waveform
                sections.append(i)
    if waveform_state != WaveformState.FIND_WAVEFORM or len(sections) == 1:
        # We are tyring to find the tail end of the waveform but have not reached it yet
        # Or we are in the noise section and so we want to cut at the end
        sections.append(i)

    return sections


def stream_to_continuous_array_nan(
    group: Stream, t: np.ndarray, sampling_rate: float
) -> np.ndarray:
    # Initialize with NaNs to represent gaps
    data = np.full_like(t, np.nan)
    t_start = t[0]
    for tr in group:
        start_idx = int((tr.stats.starttime.timestamp - t_start) * sampling_rate)
        end_idx = start_idx + len(tr.data)
        if start_idx < 0:
            tr_data = tr.data[-start_idx:]
            start_idx = 0
        else:
            tr_data = tr.data
        if end_idx > len(data):
            tr_data = tr_data[: len(data) - start_idx]
            end_idx = len(data)
        # Insert trace data
        data[start_idx : start_idx + len(tr_data)] = tr_data
    return data


def get_section(sections: list, p_wave_guess: float, ds_npts: int):
    """
    Gets the section of the waveform that is closest to the P wave arrival time
    """
    # Get every pair of numbers in the sections list
    pairs = [
        (sections[i], sections[i + 1], i, i + 1) for i in range(0, len(sections) - 1, 2)
    ]

    min_distance = None
    closest = None

    for start, end, start_i, end_i in pairs:
        if start <= p_wave_guess <= end:
            closest = (start, end, start_i, end_i)
            break
        else:
            distance = min(abs(p_wave_guess - start), abs(p_wave_guess - end))
            if min_distance is None or distance < min_distance:
                min_distance = distance
                closest = (start, end, start_i, end_i)

    if closest is None:
        print("Error: No closest section found")

    # Now we want to expand the start and end if possible
    new_start = (
        closest[0]
        if closest[2] == 0
        else int(
            sections[closest[2] - 1] + (0 if closest[2] - 1 == 0 else ds_npts * 0.5)
        )
    )
    new_end = (
        closest[1]
        if closest[3] == len(sections) - 1
        else int(
            sections[closest[3] + 1]
            - (0 if closest[3] + 2 == len(sections) else ds_npts * 0.5)
        )
    )

    return (new_start, new_end)


def run_extract_window_step(
    step,
    mseed,
    start_time,
    ptime_est,
    ds,
    orig_start_time,
    orig_end_time,
    start_time_Ia_values,
    end_time_Ia_values,
):
    start_frozen = False
    end_frozen = False

    loc = mseed[0].stats.location
    chan = mseed[0].stats.channel[:2]

    # If we have split traces
    if len(mseed) > 3:
        # Copy the mseed
        mseed_copy = mseed.copy()
        # Detrend the mseed
        mseed_copy.detrend("demean")
        mseed_copy.detrend("linear")
        # Filter the mseed
        mseed_copy.filter("bandpass", freqmin=0.1, freqmax=40)
        # Get the unique ending channel ids
        unique_channel_endings = list(set([tr.stats.channel[-1] for tr in mseed_copy]))
        trace_indexs = []
        for unique_channel_ending in unique_channel_endings:
            # We need to group together each component with the same ending channel str e.g. HN1 the 1
            group = mseed_copy.select(
                location=loc,
                channel=f"{chan}{unique_channel_ending}",
            )
            # For each trace compute the AI and keep the highest
            for i, tr in enumerate(group):
                # Get the IA and IA_norm
                Ia, _ = get_arias_intensity_norm(tr)
                if i == 0:
                    Ia_max = Ia[-1]
                    Ia_max_index = 0
                else:
                    if Ia[-1] > Ia_max:
                        Ia_max = Ia
                        Ia_max_index = i
            trace_indexs.append(Ia_max_index)
        # Check all the groups agree
        if len(set(trace_indexs)) == 1:
            # We grab the last trace from the group and trim the mseed to the start and end time
            tr = group[trace_indexs[0]]
            mseed = mseed.trim(tr.stats.starttime, tr.stats.endtime)
        else:
            # We can't agree on the trace index, skip this mseed
            return None, None, None, None, start_time_Ia_values, end_time_Ia_values

    # Ensure we crop the mseed to the max and min start and endtime
    starttime_trim = max([tr.stats.starttime for tr in mseed])
    endtime_trim = min([tr.stats.endtime for tr in mseed])
    # Check that the start time is before the end time

    if starttime_trim > endtime_trim:
        # WE NEED TO RAISE AN ERROR HERE
        return None, None, None, None, start_time_Ia_values, end_time_Ia_values
    mseed.trim(starttime_trim, endtime_trim)

    # Copy the mseed
    mseed_copy = mseed.copy()
    # Detrend the mseed
    mseed_copy.detrend("demean")
    mseed_copy.detrend("linear")
    # Filter the mseed
    mseed_copy.filter("bandpass", freqmin=0.1, freqmax=40)

    # Calculate AI for the horizontal components
    Ia, Ia_norm = get_arias_intensity_norm(mseed_copy[0])
    _, Ia_norm_1 = get_arias_intensity_norm(mseed_copy[1])

    dt = mseed[0].stats.delta
    npts = mseed[0].stats.npts

    # compute the derivative of the IA_norm
    Ia_norm_diff = np.gradient(Ia_norm[:, 1], dt)
    Ia_norm_diff_1 = np.gradient(Ia_norm_1[:, 1], dt)

    # Apply Gaussian filter to IA_norm_diff
    Ia_norm_diff = gaussian_filter1d(Ia_norm_diff, sigma=10)
    Ia_norm_diff_1 = gaussian_filter1d(Ia_norm_diff_1, sigma=10)

    final_mseed = True

    # Calculate n_points per second
    npts_per_sec = 1 / dt

    # Do the first step of splitting the waveform
    ds_npts = max((ds * 0.30), 2.5) * npts_per_sec

    if step == 1:
        sections = get_sections(ds_npts, Ia_norm_diff, Ia_norm)
        sections_1 = get_sections(ds_npts, Ia_norm_diff_1, Ia_norm_1)

        # Manage merging of the sections
        if len(sections) > len(sections_1):
            # We want the actual selections to be the higher of the two
            final_sections = sections
        elif len(sections_1) > len(sections):
            # We want the actual selections to be the higher of the two
            final_sections = sections_1
        else:
            # We go through each selection and grab the tightest selections from the 2
            final_sections = [0]
            signal = True
            for i in range(1, len(sections)):
                if signal:
                    # We are in the signal section
                    # Grab the section that is the highest
                    final_sections.append(max(sections[i], sections_1[i]))
                    signal = False
                else:
                    # We are in the noise section
                    # Grab the section that is the lowest
                    final_sections.append(min(sections[i], sections_1[i]))
                    signal = True

        # Get the section of the waveform that is closest to the P wave arrival time
        p_wave_guess = (ptime_est - start_time) * npts_per_sec
        section = get_section(final_sections, p_wave_guess, ds_npts)
    else:
        # Create the section as the start and end of the waveform
        section = (0, len(mseed[0].data) - 1)

    # Now we cut to the section
    # First we create a UTCDatetime object from the start and end time
    timestamps = np.linspace(
        mseed[0].stats.starttime.timestamp, mseed[0].stats.endtime.timestamp, npts
    )
    utc_array = [UTCDateTime(ts) for ts in timestamps]
    # Get the points for the section
    start_time = utc_array[section[0]]
    end_time = utc_array[section[1]]

    # Trim the mseed to the start and end time
    mseed.trim(start_time, end_time)

    # We check if the new start time is after the original start time, by at least ds_npts
    if start_time > orig_start_time + ds_npts / npts_per_sec:
        # We want to force the start to be frozen as we don't want to expand anymore
        # as we would go into another waveform
        start_frozen = True
    # We check if the new end time is before the original end time, by at least ds_npts
    if end_time < orig_end_time - ds_npts / npts_per_sec:
        # We want to force the end to be frozen as we don't want to expand anymore
        # as we would go into another waveform
        end_frozen = True

    if start_frozen and end_frozen:
        # We can't adjust the start and end times anymore
        final_mseed = True
    else:
        # Copy the mseed
        mseed_copy = mseed.copy()
        # Detrend the mseed
        mseed_copy.detrend("demean")
        mseed_copy.detrend("linear")
        # Filter the mseed
        mseed_copy.filter("bandpass", freqmin=0.1, freqmax=40)

        # See if we can adjust the start and end times
        # Re-calculate the IA_norm
        Ia, Ia_norm = get_arias_intensity_norm(mseed_copy[0])
        _, Ia_norm_1 = get_arias_intensity_norm(mseed_copy[1])

        # A quick check that the check value is less than the length of the Ia_norm / 2
        if ds_npts > len(Ia_norm) / 2:
            # We want to try and push the start time back
            start_value = 0.01
            start_value_1 = 0.01
            end_value = 0.96
            end_value_1 = 0.96
        else:
            # Get the value from IA_norm
            start_value = Ia_norm[int(ds_npts), 1]
            start_value_1 = Ia_norm_1[int(ds_npts), 1]
            end_value = Ia_norm[int(len(Ia_norm) - ds_npts), 1]
            end_value_1 = Ia_norm_1[int(len(Ia_norm) - ds_npts), 1]

        if (start_value > 0.02 or start_value_1 > 0.02) and not start_frozen:
            # push the start time back ds_npts / npts_per_sec
            if len(start_time_Ia_values) < 6:
                # Add the start time to the start_time_Ia_values
                start_time_Ia_values.append(
                    (start_time, max(start_value, start_value_1))
                )
                start_time -= ds_npts / npts_per_sec
                final_mseed = False
            else:
                # We have reached the limit
                # We want to take the start_time of the lowest value
                lowest_values = sorted(start_time_Ia_values, key=lambda x: x[1])
                mseed.trim(lowest_values[0][0], end_time)
                start_time = lowest_values[0][0]
                final_mseed = True

        if (end_value < 0.97 or end_value_1 < 0.97) and not end_frozen:
            if len(end_time_Ia_values) < 6:
                # Add the end time to the end_time_Ia_values
                end_time_Ia_values.append((end_time, min(end_value, end_value_1)))
                # push the end time forward ds_npts / npts_per_sec
                end_time += ds_npts / npts_per_sec
                final_mseed = False
            else:
                # We have reached the limit
                # We want to take the end_time of the highest value
                highest_values = sorted(end_time_Ia_values, key=lambda x: x[1])
                mseed.trim(start_time, highest_values[-1][0])
                end_time = highest_values[-1][0]
                final_mseed = True
    return (
        final_mseed,
        start_time,
        end_time,
        mseed,
        start_time_Ia_values,
        end_time_Ia_values,
    )


def get_waveforms(
    preferred_origin: Origin,
    client: FDSN_Client,
    net: str,
    sta: str,
    mag: float,
    rrup: float,
    r_epi: float,
    vs30: float = None,
    only_record_ids: list[str] = None,
    event_id: str = None,
):
    """
    Get the waveforms for a given event and station
    Calculate the start and end times for the waveform based on the P and S arrival times

    Parameters
    ----------
    preferred_origin : Origin
        The preferred origin object from the event catalogue
    client : FDSN_Client
        The FDSN client to use to get the waveforms
    net : str
        The network code
    sta : str
        The station
    mag : float
        The magnitude of the event
    rrup : float
        The closest distance to the event
    r_epi : float
        The epicentral distance to the event
    vs30 : float, optional
        The Vs30 value for the station, by default sets to config value
    only_record_ids : list[str], optional
        A list of record ids to get the waveforms for, by default None

    Returns
    -------
    Union[Stream, None]
        The stream object containing the waveform data or None if no data is found
    """
    config = cfg.Config()
    vs30 = config.get_value("vs30") if vs30 is None else vs30
    rake = 90  # Assume strike-slip for now
    z1p0 = z_model_calculations.chiou_young_08_calc_z1p0(vs30)
    # Predict significant duration time from Afshari and Stewart (2016)
    input_df = pd.DataFrame(
        {
            "mag": [mag],
            "rake": [rake],
            "rrup": [rrup],
            "vs30": [vs30],
            "z1pt0": [z1p0],
        }
    )
    result_df = openquake_wrapper_vectorized.oq_run(
        classdef.GMM.AS_16,
        classdef.TectType.ACTIVE_SHALLOW,
        input_df,
        "Ds595",
    )
    ds = np.exp(result_df["Ds595_mean"].values[0])

    deg = kilometers2degrees(r_epi)

    model = TauPyModel(model="iasp91")

    # Estimate arrival times for P and S phases
    p_arrivals = model.get_travel_times(
        source_depth_in_km=preferred_origin.depth / 1000,
        distance_in_degree=deg,
        phase_list=["ttp"],
    )
    s_arrivals = model.get_travel_times(
        source_depth_in_km=preferred_origin.depth / 1000,
        distance_in_degree=deg,
        phase_list=["tts"],
    )
    ptime_est = (
        preferred_origin.time + p_arrivals[0].time
    )  # Estimated earliest P arrival time from taup
    stime_est = preferred_origin.time + s_arrivals[0].time

    # Find the start and end times for the mseed data
    min_time_difference = config.get_value("min_time_difference")
    ds_multiplier = config.get_value("ds_multiplier")
    start_time = ptime_est - min_time_difference
    end_time = stime_est + (
        min_time_difference
        if stime_est + ds * ds_multiplier - ptime_est < min_time_difference
        else ds * ds_multiplier
    )
    channel_codes = ",".join(config.get_value("channel_codes"))
    location = "*"

    # Check what channel codes and locations to use from only_record_ids if provided
    if only_record_ids is not None:
        # Get the channel and location to use
        channel_codes = ",".join(
            only_record_ids["record_id"].str.split("_").str[-2].unique() + "?"
        )
        location = ",".join(
            only_record_ids["record_id"].str.split("_").str[-1].unique()
        )

    # Get the waveforms with multiple retries when IncompleteReadError occurs
    max_retries = 3
    final_mseeds = []
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

                # Get the unique channels (Using first 2 keys) and locations
                unique_channels = set(
                    [(tr.stats.channel[:2], tr.stats.location) for tr in st]
                )

                # First split the stream into the diffeent unique records
                mseeds, _ = split_stream_into_mseeds(st, unique_channels, event_id)

                for mseed in mseeds:

                    # Check for any of the channels are all 0's
                    if any(np.all(np.isclose(tr.data, 0.0)) for tr in mseed):
                        # If so, skip this mseed
                        continue

                    final_mseed = False
                    start_time_Ia_values = []
                    end_time_Ia_values = []
                    orig_start_time = start_time
                    orig_end_time = end_time
                    step = 0

                    loc = mseed[0].stats.location
                    chan = mseed[0].stats.channel[:2]

                    while not final_mseed:
                        step += 1

                        # Run the extract window step
                        (
                            final_mseed,
                            start_time,
                            end_time,
                            mseed,
                            start_time_Ia_values,
                            end_time_Ia_values,
                        ) = run_extract_window_step(
                            step,
                            mseed,
                            start_time,
                            ptime_est,
                            ds,
                            orig_start_time,
                            orig_end_time,
                            start_time_Ia_values,
                            end_time_Ia_values,
                        )
                        if final_mseed is None:
                            # If the mseed is None, we want to break out of the loop
                            break

                        if not final_mseed:
                            mseed = client.get_waveforms(
                                net,
                                sta,
                                loc,
                                f"{chan}?",
                                start_time,
                                end_time,
                                attach_response=True,
                            )
                        else:
                            final_mseeds.append(mseed)
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
            import traceback

            print(e)
            traceback.print_exc()
            return None
    return final_mseeds


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

        mseeds.append(st_new)

    return mseeds, raised_issues


def write_stream_to_mseed(stream: Stream, output_file: Path):
    """
    Write an ObsPy Stream object to a MiniSEED file using mseedlib.

    Parameters
    ----------
    stream : obspy.core.stream.Stream
        The Stream object to write to MiniSEED
    output_file : Path
        The path to the output MiniSEED file

    Raises
    ------
    ValueError
        If the sample type of the trace data is not supported
    """
    mstl = mseedlib.MSTraceList()
    for trace in stream:
        start_time = mseedlib.timestr2nstime(f"{trace.stats.starttime.isoformat()}Z")
        sourceid = f"FDSN:{trace.stats.network}_{trace.stats.station}_{trace.stats.location}_{'_'.join(trace.stats.channel)}"
        mstl.add_data(
            sourceid=sourceid,
            data_samples=trace.data,
            sample_type="i",
            sample_rate=trace.stats.sampling_rate,
            start_time=start_time,
        )

    with open(output_file, "wb") as f:
        mstl.pack(
            lambda record, handler_data: handler_data["fh"].write(record),
            {"fh": f},
            flush_data=True,
            format_version=2,
        )


def write_mseed(mseed: Stream, event_id: str, station: str, output_directory: Path):
    """
    Write the mseed files to the output directory

    Parameters
    ----------
    mseed : Stream
        The stream object containing the waveform data
    event_id : str
        The event id which is used in the filename
    station : str
        The station code which is used in the filename
    output_directory : Path
        The directory to save the mseed files
    """
    # Get the channel and location from the first trace
    channel = mseed[0].stats.channel[:2]
    location = mseed[0].stats.location

    # Create the filename and add it to the output directory
    filename = f"{event_id}_{station}_{channel}_{location}.mseed"
    mseed_ffp = output_directory / filename
    output_directory.mkdir(exist_ok=True, parents=True)

    # Write the mseed file
    write_stream_to_mseed(mseed, mseed_ffp)
