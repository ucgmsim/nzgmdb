"""
    Contains functions for generating
    the phase arrival table
"""

import itertools
import multiprocessing
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import obspy
import pandas as pd
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.management import custom_errors, file_structure
from nzgmdb.phase_arrival import picker


def get_p_wave(data: np.ndarray, dt: int) -> int:
    """
    Get the P wave arrival time from the data

    Parameters
    ----------
    data : np.ndarray
        The waveform data from a single component
    dt : int
        The sample rate of the data

    Returns
    -------
    loc : int
        The index of the P-phase arrival if
        picker.p_phase_picker() runs successfully.
        If picker.p_phase_picker() raises an
        Exception, loc is -1.
    """
    wftype = "SM"  # Input wftype is strong motion
    loc = picker.p_phase_picker(data, dt, wftype)

    return loc


def process_mseed(mseed_file: Path) -> dict[str, Any]:
    """
    Process an mseed file and return the phase arrival data.

    Parameters
    ----------
    mseed_file : Path
        Path to the mseed file.

    Returns
    -------
    new_data : dict
        A dictionary containing the information
        to write as a line in the phase arrival table.
        the dictionary keys are the phase arrival
        table's headers.
    """

    mseed_data = obspy.read(str(mseed_file))

    try:
        p_comp_000 = get_p_wave(mseed_data[0].data, mseed_data[0].stats.delta)
    except:
        print(f"picker failed on {str(mseed_file)} p_comp_000")
        p_comp_000 = -1
    try:
        p_comp_090 = get_p_wave(mseed_data[1].data, mseed_data[1].stats.delta)
    except:
        print(f"picker failed on {str(mseed_file)} p_comp_090")
        p_comp_090 = -1
    try:
        p_comp_ver = get_p_wave(mseed_data[2].data, mseed_data[2].stats.delta)
    except:
        print(f"picker failed on {str(mseed_file)} p_comp_ver")
        p_comp_ver = -1

    if p_comp_ver > 1:
        # Vertical is greater than 1 use this for the record
        p_wave_loc = p_comp_ver
    else:
        # Find the max between all three components and take away 0.5
        p_wave_loc = max(p_comp_000, p_comp_090, p_comp_ver) - 0.5

    if p_wave_loc >= 0:
        stats = mseed_data[0].stats

        new_data = {
            "evid": file_structure.get_event_id_from_mseed(mseed_file),
            "datetime": stats.starttime + timedelta(seconds=p_wave_loc),
            "net": stats.network,
            "sta": stats.station,
            "loc": stats.location,
            "chan": stats.channel[:2],
            "phase": "P",
            "t_res": None,
        }

        return new_data


def fetch_geonet_phases(mseed_file: Path) -> list[dict[str, Any]]:
    """
    Fetch the phase arrival times from Geonet for a given mseed file.

    Parameters
    ----------
    mseed_file: Path
        Path to the mseed file.

    Returns
    -------
    phase_table_entries: list[dict[str, any]]
        A list of phase arrival times.

    Raises
    ------
    InvalidNumberOfGeonetPicksException
        If more than two phase picks from Geonet match
        a given mseed file as there should only be one
        P phase pick and sometimes one S phase pick.
    """
    # Get the event ID (evid) from the mseed file
    evid = file_structure.get_event_id_from_mseed(mseed_file)

    # Read the mseed
    mseed = obspy.read(str(mseed_file))

    # Fetch all records relating to the given event ID (evid)
    # from Geonet. Fetched records include all phase picks
    # and arrival times for all combinations of
    # network, station, location, and channel
    client_NZ = FDSN_Client("GEONET")
    event = client_NZ.get_events(eventid=evid)[0]

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

    # Get arrival data corresponding to the given mseed file by matching pick_id
    mseed_arrival_pick_pairs = [
        (arrival, pick)
        for arrival in event.preferred_origin().arrivals
        for pick in picks_matching_mseed
        if pick.resource_id == arrival.pick_id
    ]

    if len(mseed_arrival_pick_pairs) > 2:
        p_count = 0
        s_count = 0
        new_picks = []
        # Make sure that both the P and S Waves only have one pick
        for arrival, pick in mseed_arrival_pick_pairs:
            if arrival.phase == "P":
                if p_count == 0:
                    new_picks.append((arrival, pick))
                p_count += 1
            if arrival.phase == "S":
                if s_count == 0:
                    new_picks.append((arrival, pick))
                s_count += 1
        mseed_arrival_pick_pairs = new_picks

    phase_table_entries = [
        {
            "evid": evid,
            "datetime": pick.time,
            "net": mseed[0].stats.network,
            "sta": mseed[0].stats.station,
            "loc": mseed[0].stats.location,
            "chan": mseed[0].stats.channel[:2],
            "phase": mseed_arrival.phase,
            "t_res": mseed_arrival.time_residual,
        }
        for mseed_arrival, pick in mseed_arrival_pick_pairs
    ]

    return phase_table_entries


def generate_phase_arrival_table(
    main_dir: Path, output_dir: Path, n_procs: int, full_output: bool = False
):
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
    full_output: bool, optional
        If True, writes an additional table that
        contains all phase arrivals from both
        picker and Geonet
    """

    # Find all mseed files recursively
    mseed_files = list(main_dir.glob("**/*.mseed"))

    with multiprocessing.Pool(processes=n_procs) as pool:
        # Map the reading function to the file list.
        # process_mseed could return None, so we filter this out.
        picker_phases_df = pd.DataFrame(
            pd.Series(pool.map(process_mseed, mseed_files)).dropna().to_list()
        )

    # Change picker_phases_df[t_res] from nan to 0.0 so Geonet t_res values
    # will not be substituted for the missing picker_phases_df[t_res] values
    picker_phases_df["t_res"] = 0.0

    # Get the Geonet phases
    # fetch_geonet_phases can return a list of length 0, 1, or 2 so
    # itertools.chain.from_iterable is used to flatten the list of lists
    with multiprocessing.Pool(processes=n_procs) as pool:
        geonet_phases_df = pd.DataFrame(
            itertools.chain.from_iterable(pool.map(fetch_geonet_phases, mseed_files))
        )

    # Check length of the geonet_phases_df
    if len(geonet_phases_df) > 0:
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
    else:
        merged_df = picker_phases_df

    # Save the phase arrival table
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(
        output_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE, index=False
    )

    if full_output:
        # Adding labels to the DataFrame columns so they
        # can be distinguished after the outer merge
        picker_phases_df_new_index.columns = (
            picker_phases_df_new_index.columns + f"_picker"
        )
        geonet_phases_df_new_index.columns = (
            geonet_phases_df_new_index.columns + f"_geonet"
        )

        all_picker_and_geonet_df = pd.merge(
            left=picker_phases_df_new_index,
            right=geonet_phases_df_new_index,
            left_index=True,
            right_index=True,
            how="outer",
        )
        all_picker_and_geonet_df = all_picker_and_geonet_df.reset_index()
        all_picker_and_geonet_df["picker_time_minus_geonet_time_secs"] = np.nan

        condition_not_nan = (
            all_picker_and_geonet_df["datetime_picker"].notnull()
            & all_picker_and_geonet_df["datetime_geonet"].notnull()
        )

        all_picker_and_geonet_df.loc[
            condition_not_nan, "picker_time_minus_geonet_time_secs"
        ] = (
            all_picker_and_geonet_df.loc[condition_not_nan, "datetime_picker"]
            - all_picker_and_geonet_df.loc[condition_not_nan, "datetime_geonet"]
        ).astype(
            np.float64
        )

        all_picker_and_geonet_df.to_csv(
            output_dir / "full_phase_arrival_table.csv", index=False
        )
