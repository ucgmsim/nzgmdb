"""
This module contains functions for creating mseed files from the waveform data from the FDSN client
"""

import http
import http.client
import time
import warnings
from collections.abc import Iterable
from pathlib import Path

import mseedlib
import numpy as np
import pandas as pd
from obspy import Trace, Stream
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
    for attempt in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)

                final_st = False
                end_time_repeats = 0
                start_time_repeats = 0
                orig_start_time = start_time
                orig_end_time = end_time

                while not final_st:
                    st = client.get_waveforms(
                        net,
                        sta,
                        location,
                        channel_codes,
                        start_time,
                        end_time,
                        attach_response=True,
                    )

                    start_frozen = False
                    end_frozen = False
                    # Checks for if the start and end time are outside the current bounds of the waveform
                    if st[0].stats.starttime > start_time:
                        start_frozen = True
                    if st[0].stats.endtime < end_time:
                        end_frozen = True
                    if start_frozen and end_frozen:
                        # We can't adjust the start and end times anymore
                        break

                    # Copy the st
                    st_copy = st.copy()
                    # Detrend the stream
                    st_copy.detrend("demean")
                    st_copy.detrend("linear")
                    # Filter the stream
                    st_copy.filter("bandpass", freqmin=0.1, freqmax=40)

                    # Calculate AI
                    Ia, Ia_norm = get_arias_intensity_norm(st_copy[0])
                    _, Ia_norm_1 = get_arias_intensity_norm(st_copy[1])

                    dt = st[0].stats.delta
                    npts = st[0].stats.npts
                    t = np.linspace(0, (npts - 1) * dt, npts)

                    # compute the derivative of the IA_norm
                    Ia_norm_diff = np.gradient(Ia_norm[:, 1], dt)
                    Ia_norm_diff_1 = np.gradient(Ia_norm_1[:, 1], dt)

                    # Apply Gaussian filter to IA_norm_diff
                    Ia_norm_diff = gaussian_filter1d(Ia_norm_diff, sigma=10)
                    Ia_norm_diff_1 = gaussian_filter1d(Ia_norm_diff_1, sigma=10)

                    # Plot the AI

                    data = st[0].data
                    data_1 = st[1].data
                    orig_start_time_sec = orig_start_time - start_time
                    orig_end_time_sec = orig_end_time - start_time

                    fig, axs = plt.subplots(
                        3,
                        2,
                        figsize=(14, 9),
                        sharex="row",
                        gridspec_kw={"hspace": 0.3},
                    )

                    # Left column = original input
                    # Right column = second input to compare

                    # --- COLUMN 1 (Original channel) ---
                    # 1. Arias Intensity
                    axs[0, 0].plot(t, Ia_norm[:, 1], label="Arias Intensity")
                    axs[0, 0].set_ylabel("Arias Intensity")
                    axs[0, 0].set_title(f"Channel 1")
                    axs[0, 0].grid(ls=":")
                    axs[0, 0].legend()

                    # 2. Derivative
                    axs[1, 0].plot(t, Ia_norm_diff, label="d(AI)/dt", color="tab:green")
                    axs[1, 0].axhline(
                        0.01, color="r", linestyle="--", label="Threshold"
                    )
                    axs[1, 0].set_ylabel("Derivative")
                    axs[1, 0].set_ylim(0, 0.05)
                    axs[1, 0].grid(ls=":")
                    axs[1, 0].legend()

                    # 3. Waveform
                    axs[2, 0].plot(t, data, label="Waveform", color="tab:orange")
                    axs[2, 0].axvline(
                        ptime_est - start_time,
                        color="b",
                        linestyle="--",
                        label="Estimated P Arrival",
                    )
                    axs[2, 0].axvline(
                        orig_start_time_sec,
                        color="g",
                        linestyle="--",
                        label="Original Start",
                    )
                    axs[2, 0].axvline(
                        orig_end_time_sec,
                        color="r",
                        linestyle="--",
                        label="Original End",
                    )
                    axs[2, 0].set_xlabel("Time [s]")
                    axs[2, 0].set_ylabel("Amplitude")
                    axs[2, 0].grid(ls=":")
                    axs[2, 0].legend()

                    # --- COLUMN 2 (Second channel/input) ---
                    # Replace these with your second channel inputs
                    # Variables you need: t2, Ia_norm2, Ia_norm_diff2, data2, ptime_est2, etc.

                    axs[0, 1].plot(t, Ia_norm_1[:, 1], label="Arias Intensity")
                    axs[0, 1].set_title("Channel 2")
                    axs[0, 1].grid(ls=":")
                    axs[0, 1].legend()

                    axs[1, 1].plot(
                        t, Ia_norm_diff_1, label="d(AI)/dt", color="tab:green"
                    )
                    axs[1, 1].axhline(
                        0.01, color="r", linestyle="--", label="Threshold"
                    )
                    axs[1, 1].set_ylim(0, 0.05)
                    axs[1, 1].grid(ls=":")
                    axs[1, 1].legend()

                    axs[2, 1].plot(t, data_1, label="Waveform", color="tab:orange")
                    axs[2, 1].axvline(
                        ptime_est - start_time,
                        color="b",
                        linestyle="--",
                        label="Estimated P Arrival",
                    )
                    axs[2, 1].axvline(
                        orig_start_time_sec,
                        color="g",
                        linestyle="--",
                        label="Original Start",
                    )
                    axs[2, 1].axvline(
                        orig_end_time_sec,
                        color="r",
                        linestyle="--",
                        label="Original End",
                    )
                    axs[2, 1].set_xlabel("Time [s]")
                    axs[2, 1].grid(ls=":")
                    axs[2, 1].legend()

                    # Optional: shared suptitle
                    fig.suptitle(
                        f"Waveform Info | Mag={mag:.2f}, Ds595={ds:.2f}, Station={sta}",
                        fontsize=12,
                    )

                    # Final layout
                    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle

                    # Create subplots with shared x-axis
                    # fig, (ax1, ax2, ax3) = plt.subplots(
                    #     3, 1, sharex=True, figsize=(10, 8)
                    # )
                    #
                    # # Plot Arias Intensity on the first subplot
                    # ax1.plot(t, Ia_norm[:, 1], label="Arias Intensity")
                    # ax1.set_ylabel("Arias Intensity")
                    # # Format title
                    # title = (
                    #     f"Waveform Info: Mag={mag:.2f}, Ds595={ds:.2f}, Station={sta}"
                    # )
                    # ax1.set_title(title)
                    # ax1.grid(ls=":")
                    # ax1.legend()
                    #
                    # # 2. Gradient of Arias Intensity
                    # ax2.plot(
                    #     t,
                    #     Ia_norm_diff,
                    #     label="d(AI)/dt",
                    #     color="tab:green",
                    # )
                    # ax2.axhline(
                    #     0.01,
                    #     color="r",
                    #     linestyle="--",
                    #     label="Threshold",
                    # )
                    # ax2.set_ylabel("Gradient")
                    # ax2.grid(ls=":")
                    # # Scale to be between 0 and 0.05
                    # ax2.set_ylim(0, 0.05)
                    # ax2.legend()
                    #
                    # # Plot waveform on the second subplot
                    # ax3.plot(t, data, label="Waveform", color="tab:orange")
                    # # Plot the ptime_est
                    # ax3.axvline(
                    #     ptime_est - start_time,
                    #     color="b",
                    #     linestyle="--",
                    #     label="Estimated P Arrival Time",
                    # )
                    # # Only plot these if the start and end times are not the same
                    # ax3.axvline(
                    #     orig_start_time_sec,
                    #     color="g",
                    #     linestyle="--",
                    #     label="Original Start Time",
                    # )
                    # ax3.axvline(
                    #     orig_end_time_sec,
                    #     color="r",
                    #     linestyle="--",
                    #     label="Original End Time",
                    # )
                    # ax3.set_xlabel("Time [s]")
                    # ax3.set_ylabel("Amplitude")
                    # ax3.grid(ls=":")
                    # ax3.legend()
                    #
                    # # Adjust layout
                    # plt.tight_layout()
                    # plt.show()

                    # save the fig
                    output_dir = (
                        "/home/joel/local/gmdb/testing_folder/new_window_10_scenarios"
                    )
                    output_file = f"{output_dir}/{(st[0].id).replace('.', '_')}.png"
                    plt.savefig(output_file)

                    plt.close()

                    final_st = True

                    # Calculate n_points per second
                    # npts_per_sec = 1 / dt
                    #
                    # # Do the first step of splitting the waveform
                    # ds_npts = (ds / 10) * npts_per_sec
                    # counter = 0
                    #
                    # check_value = npts_per_sec * 7.5
                    #
                    # # A quick check that the check value is less than the length of the Ia_norm
                    # if check_value > len(Ia_norm):
                    #     value = 0.01  # We want to try and push the start time back
                    # else:
                    #     # Get the value from IA_norm
                    #     value = Ia_norm[int(check_value), 1]
                    #
                    # if value > 0.02 and not start_frozen:
                    #     # push the start time back 5 seconds
                    #     if start_time_repeats < 6:
                    #         start_time -= 5
                    #         start_time_repeats += 1
                    #         final_st = False
                    #
                    # # A quick check that the check value is less than the length of the Ia_norm
                    # if check_value > len(Ia_norm):
                    #     value = 0.94  # We want to try and push the end time forward
                    # else:
                    #     # Get the value from IA_norm
                    #     value = Ia_norm[int(len(Ia_norm) - check_value), 1]
                    #
                    # if value < 0.97:
                    #     ds_npts = (ds / 10) * npts_per_sec
                    #     counter = 0
                    #     back_test_cut_point = None
                    #     # Travel backwards in time from end of Ia_norm_diff
                    #     for i, Ia_diff_value in enumerate(Ia_norm_diff[::-1]):
                    #         # Check if the value is less than 0.01
                    #         if Ia_diff_value < 0.01:
                    #             # Increment the counter
                    #             counter += 1
                    #         else:
                    #             # Reset the counter
                    #             counter = 0
                    #         if counter >= ds_npts:
                    #             # Check the current IA norm value that it is above 0.2
                    #             # (There is another waveform before)
                    #             ix = len(Ia_norm_diff) - i
                    #             if Ia_norm[ix, 1] > 0.2:
                    #                 # We have found the end time
                    #                 # As there is a ds_npts worth of points with less than 0.01 in a row
                    #                 back_test_cut_point = ix + counter
                    #             break
                    #
                    #     if back_test_cut_point is not None:
                    #         end_time = back_test_cut_point
                    #         end_time_repeats = 0
                    #         final_st = False
                    #     elif not end_frozen and end_time_repeats < 6:
                    #         # push the end time forward 5 seconds
                    #         end_time += 5
                    #         end_time_repeats += 1
                    #         final_st = False

                    # Flowchart is as follows
                    """
                    Check the current AI norm, check values at start and end from some value in to see the value
                    less than or greater than to find the flat points.
                    If not then we push forward or backward by some exta time
                    we check then for ledges to end the waveform extraction on, or start it
                    keep repeating till a limit is reached in either direction.
                    
                    Compare against a dataset of quality waveforms to ensure we dont screw them up
                    and new ones from the issued set to fix multiple earthquakes.
                    Do a full run to see the differences.
                    """

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
