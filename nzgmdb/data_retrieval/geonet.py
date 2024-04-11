"""
    Functions to manage Geonet Data
"""

import io
import requests
import datetime
from typing import List
from pathlib import Path
from functools import partial

import obspy as op
import numpy as np
import pandas as pd
import multiprocessing
from scipy.interpolate import interp1d
from obspy.core.inventory import Inventory
from obspy.core.event import Event, Magnitude
from obspy.geodetics import kilometers2degrees
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.management.config import Config
from nzgmdb.management import file_structure
from nzgmdb.mseed_management.creation import get_waveforms, create_mseed_from_waveforms


def get_max_magnitude(magnitudes: List[Magnitude], mag_type: str):
    """
    Helper function to get the maximum magnitude of a certain type

    Parameters
    ----------
    magnitudes : list[Magnitude]
        The list of magnitudes to search through
    mag_type : str
        The magnitude type to search for
    """
    filtered_mags = [
        mag for mag in magnitudes if mag.magnitude_type.lower() == mag_type
    ]
    if filtered_mags:
        return max(filtered_mags, key=lambda mag: mag.station_count)
    return None


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

    # Get the preferred magnitude and preferred origin
    preferred_origin = event_cat.preferred_origin()
    preferred_magnitude = event_cat.preferred_magnitude()

    # Extract basic info from the catalog
    ev_datetime = preferred_origin.time
    ev_lat = preferred_origin.latitude
    ev_lon = preferred_origin.longitude
    ev_depth = preferred_origin.depth / 1000
    ev_loc_type = str(preferred_origin.method_id).split("/")[1]
    ev_loc_grid = str(preferred_origin.earth_model_id).split("/")[1]
    ev_ndef = preferred_origin.quality.used_phase_count
    ev_nsta = preferred_origin.quality.used_station_count
    std = preferred_origin.quality.standard_error

    pref_mag_type = preferred_magnitude.magnitude_type

    if pref_mag_type.lower() == "m":
        # Get the maximum magnitude of each type
        pref_mag_type = "ML"
        magnitudes = event_cat.magnitudes
        mb_mag = get_max_magnitude(magnitudes, "mb")
        ml_loc_mag = get_max_magnitude(magnitudes, "ml")
        mlv_loc_mag = get_max_magnitude(magnitudes, "mlv")

        if mb_mag:
            # For events with few Mb measures, perform some checks.
            if mb_mag.station_count < 3:
                loc_mag = max(
                    [ml_loc_mag, mlv_loc_mag],
                    key=lambda mag: (mag.station_count if mag else 0),
                )
                if 0 < loc_mag.station_count <= mb_mag.station_count:
                    loc_mag = mb_mag
                    pref_mag_type = "Mb"
            else:
                loc_mag = mb_mag
                pref_mag_type = "Mb"
        else:
            if ml_loc_mag or mlv_loc_mag:
                loc_mag = max(
                    [ml_loc_mag, mlv_loc_mag],
                    key=lambda mag: (mag.station_count if mag else 0),
                )
            else:
                # No ML or MLv, try M
                loc_mag = next(
                    (
                        mag
                        for mag in event_cat.magnitudes
                        if mag.magnitude_type.lower() == "m"
                    ),
                    None,
                )

        # Set the preferred magnitude variables
        pref_mag = loc_mag.mag
        pref_mag_unc = loc_mag.mag_errors.uncertainty
        pref_mag_nmag = len(loc_mag.station_magnitude_contributions)
    else:
        # Set the preferred magnitude variables
        pref_mag = preferred_magnitude.mag
        pref_mag_unc = preferred_magnitude.mag_errors.uncertainty
        pref_mag_nmag = len(preferred_magnitude.station_magnitude_contributions)
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
        preferred_magnitude.mag,
        preferred_magnitude.magnitude_type,
        preferred_magnitude.mag_errors.uncertainty,
        ev_ndef,
        ev_nsta,
        pref_mag_nmag,
        std,
        reloc,
    ]

    return event_line


def get_sta_within_radius(
    event_cat: Event,
    mws: np.ndarray,
    rrups: np.ndarray,
    f_rrup: interp1d,
    inventory: Inventory,
):
    """
    Get the stations within a certain radius of the event

    Parameters
    ----------
    event_cat : Event
        The event catalog to fetch the event data from
    mws : np.ndarray
        The magnitudes from the Mw_rrup file
    rrups : np.ndarray
        The rrups from the Mw_rrup file
    f_rrup : interp1d
        The cubic interpolation function for the magnitude distance relationship
    inventory : Inventory
        The inventory of the stations from all networks to extract the stations from
    """
    preferred_magnitude = event_cat.preferred_magnitude().mag
    preferred_origin = event_cat.preferred_origin()
    event_lat = preferred_origin.latitude
    event_lon = preferred_origin.longitude

    # Get the max radius
    if preferred_magnitude < mws.min():
        rrup = np.array(rrups.min())
    elif preferred_magnitude > mws.max():
        rrup = np.array(rrups.max())
    else:
        rrup = f_rrup(preferred_magnitude)
    maxradius = kilometers2degrees(rrup)

    inv_sub = inventory.select(
        latitude=event_lat, longitude=event_lon, maxradius=maxradius
    )

    return inv_sub


def fetch_sta_mag_lines(
    event_cat: Event,
    event_id: str,
    main_dir: Path,
    client_NZ: FDSN_Client,
    client_IU: FDSN_Client,
    pref_mag_type: str,
    mws: np.ndarray,
    rrups: np.ndarray,
    f_rrup: interp1d,
):
    """
    Fetch the station magnitude lines from the geonet client to be added to the sta_mag_df
    Also creates the mseed files for the stations

    Parameters
    ----------
    event_cat : Event
        The event catalog to fetch the data from
    event_id : str
        The event id
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    client_NZ : FDSN_Client
        The geonet client to fetch the data from New Zealand
    client_IU : FDSN_Client
        The geonet client to fetch the data from the International Network (necessary for station SNZO)
    pref_mag_type : str
        The preferred magnitude type
    mws : np.ndarray
        The magnitudes from the Mw_rrup file
    rrups : np.ndarray
        The rrups from the Mw_rrup file
    f_rrup : interp1d
        The cubic interpolation function for the magnitude distance relationship
    """
    # Set constants
    config = Config()
    channel_codes = ",".join(config.get_value("channel_codes"))

    # Get the preferred_origin
    preferred_origin = event_cat.preferred_origin()
    ev_lat = preferred_origin.latitude
    ev_lon = preferred_origin.longitude

    # Get the full inventory
    inventory_NZ = client_NZ.get_stations(channel=channel_codes, level="response")
    inventory_IU = client_IU.get_stations(
        network="IU", station="SNZO", channel=channel_codes, level="response"
    )
    inventory = inventory_NZ + inventory_IU

    # Get Networks / Stations within a certain radius of the event
    inv_sub_sta = get_sta_within_radius(event_cat, mws, rrups, f_rrup, inventory)

    sta_mag_line = []
    # Loop through the Inventory Subset of Networks / Stations
    for network in inv_sub_sta:
        for station in network:
            # Get the client
            client = client_NZ if network.code == "NZ" else client_IU

            # Get the r_hyp
            dist, _, _ = op.geodetics.gps2dist_azimuth(
                ev_lat,
                ev_lon,
                station.latitude,
                station.longitude,
            )
            r_epi = dist / 1000
            ev_depth = preferred_origin.depth / 1000
            r_hyp = ((r_epi) ** 2 + (ev_depth + station.elevation) ** 2) ** 0.5

            # Get the waveforms
            st = get_waveforms(
                preferred_origin,
                client,
                network.code,
                station.code,
                event_cat.preferred_magnitude().mag,
                r_hyp,
                r_epi,
            )
            # Check that data was found
            if st is None:
                continue

            # Create the directory structure for the given event
            year = event_cat.creation_info.creation_time.year
            mseed_dir = file_structure.get_mseed_dir(main_dir, year, event_id)
            mseed_dir.mkdir(exist_ok=True, parents=True)

            # Create the mseed files
            chan_locs = create_mseed_from_waveforms(
                st, event_id, station.code, mseed_dir
            )

            for chan, loc in chan_locs:
                # Find the station magnitude
                sta_mag = next(
                    (
                        mag
                        for mag in event_cat.station_magnitudes
                        if mag.waveform_id.station_code == station.code
                        and (
                            (
                                len(mag.waveform_id.channel_code) > 2 and len(chan) > 2
                                and mag.waveform_id.channel_code[2] == chan[2]
                            )
                            or (len(chan) > 2 and chan[2] != "Z")
                        )
                    ),
                    None,
                )
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
                        network.code,
                        station.code,
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
    return sta_mag_line


def fetch_event_data(
    event_id: str,
    main_dir: Path,
    client_NZ: FDSN_Client,
    client_IU: FDSN_Client,
    mws: np.ndarray,
    rrups: np.ndarray,
    f_rrup: interp1d,
):
    """
    Fetch the event data from the geonet client to form the event and magnitude dataframes

    Parameters
    ----------
    event_id : str
        The event id to fetch the data for
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    client_NZ : FDSN_Client
        The geonet client to fetch the data from New Zealand
    client_IU : FDSN_Client
        The geonet client to fetch the data from the International Network (necessary for station SNZO)
    """

    # Get the catalog information
    cat = client_NZ.get_events(eventid=event_id)
    event_cat = cat[0]

    # Get the event line
    event_line = fetch_event_line(event_cat, event_id)

    if event_line is not None:
        # Get the station magnitude lines
        sta_mag_lines = fetch_sta_mag_lines(
            event_cat,
            event_id,
            main_dir,
            client_NZ,
            client_IU,
            event_line[8],
            mws,
            rrups,
            f_rrup,
        )
    else:
        sta_mag_lines = None
    return event_line, sta_mag_lines


def download_earthquake_data(
    start_date: datetime,
    end_date: datetime,
):
    """
    Download the earthquake data files from the geonet website
    and creates a dataframe with the data.

    Extracted into smaller requests to avoid the 20,000 event limit
    to stop their system crashing.

    Parameters
    ----------
    start_date : datetime
        The start date for the data extraction from the earthquake data
    end_date : datetime
        The end date for the data extraction from the earthquake data
    """
    # Define bbox for New Zealand
    config = Config()
    bbox = ",".join([str(coord) for coord in config.get_value("bbox")])

    # Send API request for the date ranges required
    geonet_url = config.get_value("geonet_url")
    endpoint = f"{geonet_url}/count?bbox={bbox}&startdate={start_date}&enddate={end_date}"
    response = requests.get(endpoint)

    # Check if the response is valid
    if response.status_code != 200:
        raise ValueError("Could not get the earthquake data")

    # Get the response dates
    response_json = response.json()
    # Check that the response has the "dates" key
    if "dates" not in response_json:
        response_dates = [end_date, start_date]
    else:
        response_dates = response_json["dates"]

    # Get the min and max magnitude from the config
    config = Config()
    min_mag = config.get_value("min_mag")
    max_mag = config.get_value("max_mag")

    # Loop over the dates to extract the csv data to a dataframe
    dfs = []
    for index, first_date in enumerate(response_dates[1:]):
        second_date = response_dates[index]
        endpoint = f"{geonet_url}/csv?bbox={bbox}&startdate={first_date}&enddate={second_date}"
        response = requests.get(endpoint)

        # Check if the response is valid
        if response.status_code != 200:
            raise ValueError("Could not get the earthquake data")

        # Read the response into a dataframe
        df = pd.read_csv(io.StringIO(response.text))
        # Filter the data based on magnitude
        df = df[(df["magnitude"] >= min_mag) & (df["magnitude"] <= max_mag)]

        dfs.append(df)

    # Concatenate the dataframes and sort by origintime
    geonet = (
        pd.concat(dfs, ignore_index=True)
        .sort_values("origintime")
        .reset_index(drop=True)
    )
    # Convert the origintime to datetime and remove the timezone
    geonet["origintime"] = pd.to_datetime(geonet["origintime"]).dt.tz_localize(None)

    return geonet


def parse_geonet_information(
    main_dir: Path,
    start_date: datetime,
    end_date: datetime,
    n_procs: int = 1,
):
    """
    Read the geonet information and manage the fetching of more data to create the mseed files

    Parameters
    ----------
    main_dir : Path
        The main directory to the NZGMDB results (Highest level directory)
    start_date : datetime
        The start date for the data extraction from the earthquake csv
    end_date : datetime
        The end date for the data extraction from the earthquake csv
    n_procs : int (optional)
        The number of processes to run
    """
    # Get the earthquake data
    geonet = download_earthquake_data(start_date, end_date)

    # Get all event ids
    event_ids = geonet.publicid.unique()

    # Get Station Information from geonet clients
    client_NZ = FDSN_Client("GEONET")
    client_IU = FDSN_Client("IRIS")

    # Get the data_dir
    data_dir = Path(__file__).parent.parent / "data"

    mw_rrup = np.loadtxt(data_dir / "Mw_rrup.txt")
    mws = mw_rrup[:, 0]
    rrups = mw_rrup[:, 1]
    # Generate cubic interpolation for magnitude distance relationship
    f_rrup = interp1d(mws, rrups, kind="cubic")

    # Create main directory if it doesn't exist
    main_dir.mkdir(exist_ok=True, parents=True)

    # Get each of the results for the event ids
    # with multiprocessing.Pool(processes=n_procs) as pool:
    #     results = pool.map(
    #         partial(
    #             fetch_event_data,
    #             main_dir=main_dir,
    #             client_NZ=client_NZ,
    #             client_IU=client_IU,
    #             mws=mws,
    #             rrups=rrups,
    #             f_rrup=f_rrup,
    #         ),
    #         event_ids,
    #     )

    # Do a standard for loop of the commented code above
    results = []
    for event_id in event_ids:
        results.append(
            fetch_event_data(
                event_id,
                main_dir,
                client_NZ,
                client_IU,
                mws,
                rrups,
                f_rrup,
            )
        )

    # Due to uneven lengths, need to extract using a for loop
    event_data = []
    sta_mag_data = []
    for result in results:
        if result[0] is not None:
            event_data.append(result[0])
            sta_mag_data.extend(result[1])

    # Create the event df
    event_df = pd.DataFrame(
        event_data,
        columns=[
            "evid",
            "datetime",
            "lat",
            "lon",
            "depth",
            "loc_type",
            "loc_grid",
            "mag",
            "mag_type",
            "mag_method",
            "mag_unc",
            "mag_orig",
            "mag_orig_type",
            "mag_orig_unc",
            "ndef",
            "nsta",
            "nmag",
            "t_res",
            "reloc",
        ],
    )
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

    # Create the output directory for the flatfiles
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    flatfile_dir.mkdir(exist_ok=True, parents=True)

    # Save the dataframes
    event_df.to_csv(flatfile_dir / "earthquake_source_table.csv", index=False)
    sta_mag_df.to_csv(flatfile_dir / "station_magnitude_table.csv", index=False)
