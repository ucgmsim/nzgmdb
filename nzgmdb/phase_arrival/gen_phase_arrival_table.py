"""
    Contains functions for generating
    the phase arrival table
"""

import multiprocessing
from datetime import timedelta
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.management import file_structure
from .picker import p_phase_picker


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
        mseed_data = obspy.read(str(file))

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


class too_many_matching_geonet_picks_Exception(Exception):
    """
    This exception is raised if more than two phase picks
    from Geonet match a given mseed file as there should
    only be one P phase pick and sometimes one S phase pick.
    """

    pass


def fetch_geonet_phases(mseed_file: Path):
    """
    Fetch the phase arrival times from
    Geonet for a given mseed file.

    Parameters
    ----------
    mseed_file: Path
        Path to the mseed file.

    Returns
    ----------
    phase_lines_for_table: list
        A list of phase arrival times.
    """

    # Creating an empty list that will be populated and returned
    phase_lines_for_table = []

    # Get the event ID (evid) from the mseed file
    evid = file_structure.get_event_id_from_mseed(mseed_file)

    # Read the mseed
    mseed = obspy.read(str(mseed_file))

    # Fetch all records relating to the given event ID (evid)
    # from Geonet. Fetched records include all phase picks
    # and arrival times for all combinations of
    # network, station, location, and channel
    client_NZ = FDSN_Client("GEONET")
    cat = client_NZ.get_events(eventid=evid)
    event = cat[0]

    # Find the picks corresponding to the given mseed file (matching network, station, location, and channel)
    picks_matching_mseed = []
    for pick in event.picks:
        if (
            pick.waveform_id.network_code == mseed[0].stats.network
            and pick.waveform_id.station_code == mseed[0].stats.station
            and pick.waveform_id.location_code == mseed[0].stats.location
            and pick.waveform_id.channel_code[:2] == mseed[0].stats.channel[:2]
        ):
            picks_matching_mseed.append(pick)

    # Check that the number of matching picks is acceptable
    if len(picks_matching_mseed) > 2:
        raise too_many_matching_geonet_picks_Exception(
            "More than two phase picks from Geonet seem to match the given mseed file."
            "\nThere should only be one P phase pick and sometimes one S phase pick."
        )

    # Get arrival data corresponding to the given mseed file by matching pick_id
    mseed_arrival_pick_pairs = [
        (arrival, pick)
        for arrival in event.preferred_origin().arrivals
        for pick in picks_matching_mseed
        if pick.resource_id == arrival.pick_id
    ]

    # Create the lines to write in the phase arrival table
    for mseed_arrival, pick in mseed_arrival_pick_pairs:
        phase_line = (
            {
                "evid": evid,
                "datetime": pick.time,
                "net": mseed[0].stats.network,
                "sta": mseed[0].stats.station,
                "loc": mseed[0].stats.location,
                "chan": mseed[0].stats.channel[:2],
                "phase": mseed_arrival.phase,
                "t_res": mseed_arrival.time_residual,
            },
        )
        phase_lines_for_table.append(phase_line)

    return phase_lines_for_table


def generate_phase_arrival_table(main_dir: Path, output_dir: Path, n_procs: int):
    """
    Generate the phase arrival table

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

    # Create the dataframe for phases from picker
    picker_phases_df = pd.DataFrame(
        [tup[0] for data_list in mseed_data_list for tup in data_list]
    )

    # Change picker_phases_df[t_res] from nan to 0.0 so Geonet t_res values
    # will not be substituted for the missing picker_phases_df[t_res] values
    picker_phases_df["t_res"] = 0.0

    # Get the Geonet phases
    geonet_phase_lines = []
    for mseed_file in mseed_files:
        geonet_phase_lines.extend(fetch_geonet_phases(mseed_file))

    # Create a DataFrame containing Geonet phases
    geonet_phases_df = pd.DataFrame([tup[0] for tup in geonet_phase_lines])

    # Use other columns as a new DataFrame index
    columns_to_merge_for_new_index = ["evid", "net", "sta", "loc", "chan", "phase"]
    geonet_phases_df_new_index = geonet_phases_df.set_index(
        columns_to_merge_for_new_index
    )
    picker_phases_df_new_index = picker_phases_df.set_index(
        columns_to_merge_for_new_index
    )

    # Use the new index to include Geonet phase
    # arrival times if there are not any conflicting
    # phase arrival times from picker
    merged_df = picker_phases_df_new_index.combine_first(geonet_phases_df_new_index)

    # reset the index back to normal
    merged_df = merged_df.reset_index()

    # Save the phase arrival table
    merged_df.to_csv(output_dir / "phase_arrival_table.csv", index=False)
