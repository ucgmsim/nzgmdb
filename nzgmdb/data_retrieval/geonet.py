"""
    Functions to manage Geonet Data
"""

import warnings
import datetime
from pathlib import Path

import pandas as pd
import obspy as op
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.core.event import Event


def fetch_event_line(event_cat: Event, event_id: str):
    """
    Fetch the event line from the geonet client to be added to the event_df

    Parameters
    ----------
    event_cat : Event
        The event catalog to fetch the data from
    event_id : str
        The event id to add to the event line
    """
    reloc = "no"  # Indicates if an earthquake has been relocated, default to 'no'.

    # Extract basic info from the catalog
    ev_datetime = event_cat.preferred_origin().time
    ev_lat = event_cat.preferred_origin().latitude
    ev_lon = event_cat.preferred_origin().longitude
    ev_depth = event_cat.preferred_origin().depth / 1000
    try:
        ev_loc_type = str(event_cat.preferred_origin().method_id).split("/")[1]
    except:
        ev_loc_type = None
    try:
        ev_loc_grid = str(event_cat.preferred_origin().earth_model_id).split("/")[1]
    except:
        ev_loc_grid = None
    try:
        ev_ndef = event_cat.preferred_origin().quality.used_phase_count
    except:
        ev_ndef = None
    try:
        ev_nsta = event_cat.preferred_origin().quality.used_station_count
    except:
        ev_nsta = None
    try:
        std = event_cat.preferred_origin().quality.standard_error
    except:
        std = None

    def get_max_magnitude(magnitudes, mag_type):
        """
        Helper function to get the maximum magnitude of a certain type
        """
        filtered_mags = [mag for mag in magnitudes if mag.magnitude_type.lower() == mag_type]
        if filtered_mags:
            return max(filtered_mags, key=lambda mag: mag.station_count)
        return None

    pref_mag_type = event_cat.preferred_magnitude().magnitude_type
    if pref_mag_type.lower() == "m":
        # Get the maximum magnitude of each type
        pref_mag_type = "ML"
        mb_mag = get_max_magnitude(event_cat.magnitudes, "mb")
        ml_loc_mag = get_max_magnitude(event_cat.magnitudes, "ml")
        mlv_loc_mag = get_max_magnitude(event_cat.magnitudes, "mlv")

        if mb_mag:
            # For events with few Mb measures, perform some checks.
            if mb_mag.station_count < 3:
                loc_mag = max([ml_loc_mag, mlv_loc_mag], key=lambda mag: (mag.station_count if mag else 0))
                if 0 < loc_mag.station_count <= mb_mag.station_count:
                    loc_mag = mb_mag
                    pref_mag_type = "Mb"
            else:
                loc_mag = mb_mag
                pref_mag_type = "Mb"
        else:
            if ml_loc_mag or mlv_loc_mag:
                loc_mag = max([ml_loc_mag, mlv_loc_mag], key=lambda mag: (mag.station_count if mag else 0))
            else:
                # No ML or MLv, try M
                loc_mag = next((mag for mag in event_cat.magnitudes if mag.magnitude_type.lower() == "m"), None)

        # Set the preferred magnitude variables
        pref_mag = loc_mag.mag
        pref_mag_unc = loc_mag.mag_errors.uncertainty
        pref_mag_nmag = len(loc_mag.station_magnitude_contributions)
    else:
        # Set the preferred magnitude variables
        pref_mag = event_cat.preferred_magnitude().mag
        pref_mag_unc = event_cat.preferred_magnitude().mag_errors.uncertainty
        pref_mag_nmag = len(
            event_cat.preferred_magnitude().station_magnitude_contributions
        )
    pref_mag_method = "uncorrected"

    # Create the event line
    event_line = [
        event_id,
        ev_datetime,
        ev_lat,
        ev_lon,
        ev_depth,
        ev_loc_type,
        ev_loc_grid,
        pref_mag,
        pref_mag_type,
        pref_mag_method,
        pref_mag_unc,
        event_cat.preferred_magnitude().mag,
        event_cat.preferred_magnitude().magnitude_type,
        event_cat.preferred_magnitude().mag_errors.uncertainty,
        ev_ndef,
        ev_nsta,
        pref_mag_nmag,
        std,
        reloc,
    ]

    return event_line


def fetch_arrival_lines(event_cat: Event, event_id: str):
    """
    Fetch the arrival lines from the geonet client to be added to the arrival_df for the given event

    Parameters
    ----------
    event_cat : Event
        The event catalog to fetch the data from
    event_id : str
        The event id to add to the arrival line
    """

    # Get the arrivals
    arrivals = event_cat.preferred_origin().arrivals

    # Get arrival data
    phase_lines = []
    for i, arrival in enumerate(arrivals):
        arid = event_id + str("a") + str(i + 1)
        arr_t_res = arrival.time_residual
        pick = [
            pick for pick in event_cat.picks if pick.resource_id == arrival.pick_id
        ][0]
        arr_datetime = pick.time
        net = pick.waveform_id.network_code
        sta = pick.waveform_id.station_code
        loc = pick.waveform_id.location_code
        chan = pick.waveform_id.channel_code
        phase_lines.append(
            [arid, arr_datetime, net, sta, loc, chan, arrival.phase, arr_t_res, event_id]
        )

    return phase_lines


def fetch_sta_mag_lines(event_cat: Event, event_id: str, station_df: pd.DataFrame, sta_corr: pd.DataFrame, client_NZ: FDSN_Client, client_IU: FDSN_Client, pref_mag_type: str):
    """
    """
    # Set constants
    sorter = ["HH", "BH", "EH", "SH", "HN", "BN"]
    channel_filter = "HH?,BH?,EH?,SH?,HN?,BN?"

    # Get the pick data
    pick_data = [
        [
            pick.waveform_id.network_code,
            pick.waveform_id.station_code,
            pick.waveform_id.location_code,
            pick.waveform_id.channel_code,
            pick.time,
            pick.phase_hint,
            pick.resource_id,
        ]
        for pick in event_cat.picks
    ]
    amplitudes = event_cat.amplitudes
    for amp in amplitudes:
        if not any(amp.waveform_id.station_code in pick for pick in pick_data):
            pick_data.append([
                amp.waveform_id.network_code,
                amp.waveform_id.station_code,
                amp.waveform_id.location_code,
                amp.waveform_id.channel_code,
                amp.time_window.reference,
                "none",
                "none"
            ])

    sta_mag_line = []
    mag_line = []
    i = 1
    # Loop through the pick data
    for net, sta, loc, chan, time, phase_hint, resource_id in pick_data:
        # Get the client
        client = client_NZ if net == "NZ" else client_IU

        # Checks if there already is an entry for the given net and station combo
        ns = [net, sta]
        row_exists = [row[1:3] for row in sta_mag_line if row[1:3] == ns]
        if row_exists:
            continue

        # Get the Inventory st
        inventory_st = []
        search_channel = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                inventory_st = client.get_stations(
                    network=net,
                    station=sta,
                    channel=channel_filter,
                    level="response",
                    starttime=time,
                    endtime=time,
                )
                for channel_code in sorter:
                    for c in inventory_st[0][0]:
                        if c.code.startswith(channel_code):
                            loc = c.location_code
                            chan = f"{channel_code}{chan[-1]}"
                            search_channel = False
                            break
                    if not search_channel:
                        break
            except:
                pass

        # Check the length of the inventory
        if len(inventory_st) == 0:
            sta_mags = [
                sta_mag
                for sta_mag in event_cat.station_magnitudes
                if (
                        (sta_mag.waveform_id.network_code == net)
                        & (sta_mag.waveform_id.station_code == sta)
                )
            ]
            for sta_mag in sta_mags:
                amp = next((amp for amp in event_cat.amplitudes if amp.resource_id == sta_mag.amplitude_id), None)
                if amp:
                    amp_amp = amp.generic_amplitude
                    amp_unit = amp.unit if "unit" in amp else None
                else:
                    amp_amp = amp_unit = None
                magid = event_id + str("m") + str(i)
                i = i + 1
                mag_line.append(
                    [
                        magid,
                        net,
                        sta,
                        sta_mag.waveform_id.location_code,
                        sta_mag.waveform_id.channel_code,
                        event_id,
                        sta_mag.mag,
                        sta_mag.station_magnitude_type,
                        None,
                        "uncorrected",
                        amp_amp,
                        None,
                        None,
                        None,
                        amp_unit,
                        None,
                        None,
                        None,
                    ]
                )
                sta_mag_line.append(
                    [
                        magid,
                        net,
                        sta,
                        sta_mag.waveform_id.location_code,
                        sta_mag.waveform_id.channel_code,
                        event_id,
                        sta_mag.mag,
                        sta_mag.station_magnitude_type,
                        None,
                        "uncorrected",
                        amp_amp,
                        None,
                        None,
                        None,
                        amp_unit,
                        None,
                        None,
                        None,
                    ]
                )
            continue
        if (
            pref_mag_type.lower() == "ml"
            or pref_mag_type.lower() == "mlv"
            or pref_mag_type is None
        ):
            # Get the station info from the station df
            sta_info = station_df[station_df.sta == sta]

            # Get the station correction
            corr = sta_corr[sta_corr.sta == sta]["corr"].values[0] if len(sta_corr[sta_corr.sta == sta]) == 1 else 0

            # Event Info
            ev_datetime = event_cat.preferred_origin().time
            ev_lat = event_cat.preferred_origin().latitude
            ev_lon = event_cat.preferred_origin().longitude
            ev_depth = event_cat.preferred_origin().depth / 1000

            # Get the r_epi and r_hyp
            if len(sta_info) > 0:
                dist, az, b_az = op.geodetics.gps2dist_azimuth(
                    ev_lat,
                    ev_lon,
                    sta_info["lat"].values[0],
                    sta_info["lon"].values[0],
                )
                r_epi = dist / 1000
                r_hyp = (
                                r_epi ** 2
                                + (ev_depth + sta_info["elev"].values[0] / 1000) ** 2
                        ) ** 0.5
            else:
                # If the station is not in the current inventory
                # Get the arrivals
                arrivals = event_cat.preferred_origin().arrivals
                arrival = [
                    arrival
                    for arrival in arrivals
                    if arrival.pick_id == resource_id
                ][0]
                r_epi = op.geodetics.degrees2kilometers(arrival.distance)
                r_hyp = ((r_epi) ** 2 + (ev_depth) ** 2) ** 0.5

            # Get the noise and signal windows
            slow_vel = 3
            endtime = ev_datetime + r_hyp / slow_vel + 30
            windowStart = time - 35 if phase_hint.lower()[0] == "p" else time - 45
            windowEnd = endtime
            noise_window = [windowStart, windowStart + 30]
            signal_window = [time - 2, time + 30] if phase_hint.lower()[0] == "p" else [time - 12, time + 20]

            try:
                st = client.get_waveforms(net, sta, loc, chan + "?" if len(chan) < 3 else chan[0:2] + "?", windowStart,
                                          windowEnd)
                st = st.merge()
            except:
                continue

            for tr in st:
                filt_data = False

                sta_mag = None
                for sta_mag_search in event_cat.station_magnitudes:
                    if sta_mag_search.waveform_id.station_code == sta:
                        if len(sta_mag_search.waveform_id.channel_code) > 2:
                            if (
                                    sta_mag_search.waveform_id.channel_code[2]
                                    == tr.stats.channel[2]
                            ):
                                sta_mag = sta_mag_search
                                break
                        elif (
                                tr.stats.channel[2] != "Z"
                        ):  # For 2012 + data, horizontal channels are combined
                            sta_mag = sta_mag_search
                            break
                if sta_mag:
                    sta_mag_mag = sta_mag.mag
                    sta_mag_type = sta_mag.station_magnitude_type
                    amp = next((amp for amp in event_cat.amplitudes if amp.resource_id == sta_mag.amplitude_id), None)
                else:
                    sta_mag_mag = None
                    sta_mag_type = pref_mag_type
                    amp = None

                # Get the amp values
                amp_amp = amp.generic_amplitude if amp else None
                amp_unit = amp.unit if amp and "unit" in amp else None

                tr = tr.copy()
                tr = tr.split().detrend("demean").merge(fill_value=0)[0]

                tr_starttime = tr.stats.starttime
                tr_endtime = tr.stats.endtime

                tr.trim(
                    tr.stats.starttime - 5,
                    tr.stats.endtime,
                    pad=True,
                    fill_value=tr.data[0],
                )
                tr.trim(
                    tr.stats.starttime,
                    tr.stats.endtime + 5,
                    pad=True,
                    fill_value=tr.data[-1],
                )

                if tr.max() == 0:
                    if sta_mag:
                        print(f"Waveform for {tr.id} is empty")
                        magid = event_id + str("m") + str(i)
                        i = i + 1
                        mag_line.append(
                            [
                                magid,
                                net,
                                sta,
                                loc,
                                tr.stats.channel,
                                event_id,
                                sta_mag_mag,
                                sta_mag_type,
                                None,
                                "uncorrected",
                                amp_amp,
                                None,
                                None,
                                None,
                                amp_unit,
                                None,
                                None,
                                None,
                            ]
                        )
                        sta_mag_line.append(
                            [
                                magid,
                                net,
                                sta,
                                loc,
                                tr.stats.channel,
                                event_id,
                                sta_mag_mag,
                                sta_mag_type,
                                None,
                                "uncorrected",
                                amp_amp,
                                None,
                                None,
                                None,
                                amp_unit,
                                None,
                                None,
                                None,
                            ]
                        )
                    continue

    return None


def fetch_event_data(event_id: str, station_df: pd.DataFrame, sta_corr: pd.DataFrame, client_NZ: FDSN_Client, client_IU: FDSN_Client):
    """
    Fetch the event data from the geonet client to form the event, arrival and magnitude dataframes

    Parameters
    ----------
    event_id : str
        The event id to fetch the data for
    station_df : pd.DataFrame
        The dataframe containing the station information
    sta_corr : pd.DataFrame
        The dataframe containing the station corrections
    client_NZ : FDSN_Client
        The geonet client to fetch the data from New Zealand
    client_IU : FDSN_Client
        The geonet client to fetch the data from the International Network (necessary for station SNZO)
    """
    lowcut = 1
    corners = 4

    # Get the catalog information
    cat = client_NZ.get_events(eventid=event_id)
    event_cat = cat[0]

    # Get the event line
    event_line = fetch_event_line(event_cat, event_id)

    # Get the arrival lines
    arrival_lines = fetch_arrival_lines(event_cat, event_id)

    return None


def parse_geonet_information(
    eq_csv: Path,
    station_corrections_ffp: Path,
    output_dir: Path,
    start_date: datetime,
    end_date: datetime,
):
    """
    Read the geonet information and manage the fetching of more data to create the mseed files

    Parameters
    ----------
    eq_csv : Path
        The path to the earthquake csv file containing the geonet information downloaded from
        the geonet website
    station_corrections_ffp : Path
        The full file path to the station corrections file
    output_dir : Path
        The directory to save the mseed files
    start_date : datetime
        The start date for the data extraction from the earthquake csv
    end_date : datetime
        The end date for the data extraction from the earthquake csv
    """

    # Process the earthquake csv file
    geonet = pd.read_csv(eq_csv, low_memory=False)
    geonet = geonet.sort_values("origintime")
    geonet["origintime"] = geonet.origintime.apply(lambda x: UTCDateTime(x).datetime)
    geonet = geonet.reset_index(drop=True)

    # Extract the data from the geonet dataframe within the date range
    geonet = geonet[(geonet.origintime >= start_date) & (geonet.origintime <= end_date)]

    # Get all event ids
    event_ids = geonet.publicid.unique()

    # Get Station Information from geonet clients
    client_NZ = FDSN_Client("GEONET")
    client_IU = FDSN_Client("IRIS")
    inventory_NZ = client_NZ.get_stations()
    inventory_IU = client_IU.get_stations(
        network="IU", station="SNZO,AFI,CTAO,RAO,FUNA,HNR,PMG"
    )
    inventory_AU = client_IU.get_stations(network="AU")
    inventory = inventory_NZ + inventory_IU + inventory_AU

    # Extract the station information into list form for the df
    station_info = [
        [
            network.code,
            station.code,
            station.latitude,
            station.longitude,
            station.elevation,
        ]
        for network in inventory
        for station in network
    ]

    # Create the station df
    station_df = pd.DataFrame(
        station_info, columns=["net", "sta", "lat", "lon", "elev"]
    )
    station_df = station_df.drop_duplicates().reset_index(drop=True)

    # Read the station corrections file
    sta_corr = pd.read_csv(station_corrections_ffp)

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get a line for each df
    for event_id in event_ids:
        result = fetch_event_data(str(event_id), station_df, sta_corr, client_NZ, client_IU)


parse_geonet_information(
    Path("/home/joel/code/nzgmdb/nzgmdb/data/earthquakes.csv"),
    Path("/home/joel/local/gmdb/waveform_data/sta_corr.csv"),
    Path("/home/joel/local/gmdb/new_data_refactor"),
    datetime.datetime(2022, 1, 1),
    datetime.datetime(2022, 2, 1),
)
