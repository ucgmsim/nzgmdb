"""
    Contains functions for generating the phase arrival table
"""

import multiprocessing
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.management import file_structure
from .picker import p_phase_picker


# def get_p_wave(data: np.ndarray, dt: int):
#     """
#     Get the P wave arrival time from the data
#
#     Parameters
#     ----------
#     data : np.ndarray
#         The waveform data from a single component
#     dt : int
#         The sample rate of the data
#     """
#     wftype = "SM"  # Input wftype is strong motion
#     try:
#         loc = p_phase_picker(data, dt, wftype)
#     except Exception as e:
#         print(e)
#         loc = -1
#     return loc


def get_p_wave(data: np.ndarray, dt: int):
    """
    Get the P wave arrival time from the data

    Parameters
    ----------
    data : np.ndarray
        The waveform data from a single component
    dt : int
        The sample rate of the data
    """
    wftype = "SM"  # Input wftype is strong motion
    try:
        loc = p_phase_picker(data, dt, wftype)
    except Exception as e:
        print(e)
        loc = -1
    return loc


def process_mseed(mseed_file_chunk: list):
    """
    Process a chunk of mseed files and return the phase arrival data

    Parameters
    ----------
    mseed_file_chunk : list
        A list of mseed files to process
    """
    data_list = []
    for file in mseed_file_chunk:
        # Read the mseed file
        mseed_data = read(str(file))

        try:
            p_comp_000 = get_p_wave(mseed_data[0].data, mseed_data[0].stats.delta)
        except:
            p_comp_000 = -1
        try:
            p_comp_090 = get_p_wave(mseed_data[1].data, mseed_data[1].stats.delta)
        except:
            p_comp_090 = -1
        try:
            p_comp_ver = get_p_wave(mseed_data[2].data, mseed_data[2].stats.delta)
        except:
            p_comp_ver = -1

        if p_comp_ver > 1:
            # Vertical is greater than 1 use this for the record
            p_wave_loc = p_comp_ver
        else:
            # Find the max between all three components and take away 0.5
            p_wave_loc = max(p_comp_000, p_comp_090, p_comp_ver) - 0.5

        if p_wave_loc < 0:
            continue
        else:
            arid = None
            stats = mseed_data[0].stats

            # Get the datetime
            datetime = stats.starttime + timedelta(seconds=p_wave_loc)

            # Get the net
            net = stats.network

            # Get the sta
            sta = stats.station

            # Get the loc
            loc = stats.location

            # Get the chan
            chan = stats.channel

            # Get the phase
            phase = "P"

            # Get the t_res
            t_res = None

            # Get the evid
            evid = file_structure.get_event_id_from_mseed(file)

            new_data = (
                # {
                #     "arid": arid,
                #     "datetime": datetime,
                #     "net": net,
                #     "sta": sta,
                #     "loc": loc,
                #     "chan": chan,
                #     "phase": phase,
                #     "t_res": t_res,
                #     "evid": evid,
                # },
                {
                    "evid": evid,
                    "datetime": datetime,
                    "net": net,
                    "sta": sta,
                    "loc": loc,
                    "chan": chan[:2],
                    "phase": phase,
                    "t_res": t_res,
                },
            )
            data_list.append(new_data)
    return data_list


def fetch_geo_phase(mseed_file: Path):
    phase_lines = []
    # Get the evid
    evid = file_structure.get_event_id_from_mseed(mseed_file)

    # Read the mseed
    mseed = read(str(mseed_file))
    stats = mseed[0].stats

    client_NZ = FDSN_Client("GEONET")
    cat = client_NZ.get_events(eventid=evid)

    event = cat[0]
    arrivals = event.preferred_origin().arrivals

    # Get mseed info
    network = stats.network
    sta = stats.station
    loc = stats.location
    chan = stats.channel[:2]

    # Find the pick
    picks = event.picks
    mseed_picks = []
    for pick in picks:
        if (
            pick.waveform_id.network_code == network
            and pick.waveform_id.station_code == sta
            and pick.waveform_id.location_code == loc
            and pick.waveform_id.channel_code[:2] == chan
        ):
            mseed_picks.append(pick)

    # Get arrival data
    mseed_arrivals = [
        (arrival, pick)
        for arrival in arrivals
        for pick in mseed_picks
        if pick.resource_id == arrival.pick_id
    ]

    if len(mseed_picks) > 2:
        print(
            "Two primary phase arrival times for a given event ID, station, and channel were found in Geonet"
        )
        raise Exception

    added_p_wave = False
    potential_p_wave = None

    for mseed_arrival, pick in mseed_arrivals:
        phase = mseed_arrival.phase
        arr_t_res = mseed_arrival.time_residual
        if phase != "P":
            phase_line = (
                {
                    "evid": evid,
                    "datetime": pick.time,
                    "net": network,
                    "sta": sta,
                    "loc": loc,
                    "chan": chan,
                    "phase": phase,
                    "t_res": arr_t_res,
                },
            )
            phase_lines.append(phase_line)
        elif not added_p_wave:
            if pick.waveform_id.channel_code[-1] == "Z":
                phase_line = (
                    {
                        "evid": evid,
                        "datetime": pick.time,
                        "net": network,
                        "sta": sta,
                        "loc": loc,
                        "chan": chan,
                        "phase": phase,
                        "t_res": arr_t_res,
                    },
                )
                phase_lines.append(phase_line)
                added_p_wave = True
            else:
                if potential_p_wave is None:
                    potential_p_wave = (mseed_arrival, pick)
                else:
                    current_time = potential_p_wave[1].time
                    potential_new_time = pick.time

                    potential_p_wave = (
                        potential_p_wave
                        if current_time < potential_new_time
                        else (mseed_arrival, pick)
                    )

    if potential_p_wave is not None:
        phase = potential_p_wave[0].phase
        arr_t_res = potential_p_wave[0].time_residual
        phase_line = (
            {
                "evid": evid,
                "datetime": potential_p_wave[1].time,
                "net": network,
                "sta": sta,
                "loc": loc,
                "chan": chan,
                "phase": phase,
                "t_res": arr_t_res,
            },
        )
        phase_lines.append(phase_line)

    return phase_lines


def generate_phase_arrival_table(main_dir: Path, output_dir: Path, n_procs: int):
    """
    Generate the phase arrival table from a directory of mseed files
    and save it to a csv file in the output directory

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
        (glob is used to find all mseed files recursively)
    output_dir : Path
        The directory to save the phase arrival table
    n_procs : int
        The number of processes to use
    """

    # Find all mseed files recursively
    mseed_files = list(main_dir.glob("**/*.mseed"))

    # Split the mseed files into chunks based on the number of processes
    file_chunks = [mseed_files[i::n_procs] for i in range(n_procs)]

    # Initialize a multiprocessing Pool
    with multiprocessing.Pool(processes=n_procs) as pool:
        # Map the reading function to the file chunks
        mseed_data_list = pool.map(process_mseed, file_chunks)

    # Create the dataframe
    df = pd.DataFrame([tup[0] for data_list in mseed_data_list for tup in data_list])

    geo_phase_lines = []
    for mseed_file in mseed_files:
        geo_phase_lines.extend(fetch_geo_phase(mseed_file))

    # Create DataFrame
    geo_df = pd.DataFrame([tup[0] for tup in geo_phase_lines])

    # Setting all df t_res = 0.0 because if they are left as nan, the Geonet t_res values are subsituted even though
    # we are using the arrival time from picker (instead of from Geonet)
    df["t_res"] = 0.0

    geo_df_unique_label_cols_to_exclude = ["datetime", "t_res"]
    geo_df_unique_label_cols = [
        x for x in list(geo_df) if x not in geo_df_unique_label_cols_to_exclude
    ]
    geo_df_unique_label = geo_df.set_index(geo_df_unique_label_cols)
    df_unique_label = df.set_index(geo_df_unique_label_cols)

    f1 = df_unique_label
    f2 = geo_df_unique_label

    # Adds extra fields that are not already in f1 from f2 that dont have the same index names
    int_df = f1.combine_first(f2)
    # reset the index back to normal
    int_df = int_df.reset_index()

    # Save the dataframe
    df.to_csv(output_dir / "picker_phase_arrival_table.csv", index=False)

    # Save the Geonet phase arrival table
    geo_df.to_csv(output_dir / "geonet_phase_arrival_table.csv", index=False)

    # Save the dataframe with Geonet arrival times and Picker's arrival times when available
    int_df.to_csv(
        output_dir / "merged_picker_and_geonet_phase_arrival_table.csv", index=False
    )
