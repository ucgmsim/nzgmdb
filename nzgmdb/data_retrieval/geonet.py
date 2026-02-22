"""
Functions to manage Geonet Data
"""

import datetime
import functools
import io
import multiprocessing as mp
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import obspy
import pandas as pd
import requests
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.core.event import Event, Magnitude
from obspy.core.inventory import Inventory, Network, Station
from obspy.geodetics import kilometers2degrees
from obspy.taup import TauPyModel
from pandas.errors import EmptyDataError
from scipy.interpolate import interp1d

from nzgmdb.data_retrieval import inventory_xml
from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors, file_structure
from nzgmdb.management.data_registry import NZGMDB_DATA
from oq_wrapper import constants, estimations, wrapper


class EventData(NamedTuple):
    """
    A named tuple to store each events data.
    """

    event_line: list[Any]
    """The full event line with all metadata."""
    station_extraction_table: pd.DataFrame
    """The station extraction table with all metadata for wavefrom extraction."""
    skipped_records: pd.DataFrame
    """The skipped records with reasons for skipping a record"""


def get_max_magnitude(magnitudes: list[Magnitude], mag_type: str):
    """
    Helper function to get the maximum magnitude of a certain type

    Parameters
    ----------
    magnitudes : list[Magnitude]
        The list of magnitudes to search through
    mag_type : str
        The magnitude type to search for

    Returns
    -------
    float, None
        The maximum magnitude of the given type or None if not found
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
        The event catalogue to fetch the data from
    event_id : str
        The event id to add to the event line

    Returns
    -------
    list
        The event line to be added to the event_df
    """
    reloc = "no"  # Indicates if an earthquake has been relocated, default to 'no'.

    # Get the preferred magnitude and preferred origin
    preferred_origin = event_cat.preferred_origin()
    preferred_magnitude = event_cat.preferred_magnitude()

    # If the preferred origin or magnitude is None, return None
    if preferred_origin is None or preferred_magnitude is None:
        return None

    # Extract basic info from the catalogue
    ev_datetime = preferred_origin.time
    ev_lat = preferred_origin.latitude
    ev_lon = preferred_origin.longitude
    ev_depth = preferred_origin.depth / 1000
    ev_loc_type = (
        None
        if preferred_origin.method_id is None
        else str(preferred_origin.method_id).split("/")[1]
    )
    ev_loc_grid = (
        None
        if preferred_origin.earth_model_id is None
        else str(preferred_origin.earth_model_id).split("/")[1]
    )
    quality = preferred_origin.quality
    if quality is None:
        ev_ndef = None
        ev_nsta = None
        std = None
    else:
        ev_ndef = None if quality.used_phase_count is None else quality.used_phase_count
        ev_nsta = (
            None if quality.used_station_count is None else quality.used_station_count
        )
        std = None if quality.standard_error is None else quality.standard_error

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


def get_stations_within_radius(
    event_cat: Event,
    mw_rrup_data: np.ndarray,
    inventory: Inventory,
):
    """
    Get the stations within a certain radius of the event

    Parameters
    ----------
    event_cat : Event
        The event catalogue to fetch the event data from
    mw_rrup_data : np.ndarray
        The Mw_rrup data to get the interpolation function to determine the max radius.
    inventory : Inventory
        The inventory of the stations from all networks to extract the stations from

    Returns
    -------
    Inventory
        The subset of the inventory with the stations within the radius
    """
    preferred_magnitude = event_cat.preferred_magnitude().mag
    preferred_origin = event_cat.preferred_origin()
    event_lat = preferred_origin.latitude
    event_lon = preferred_origin.longitude
    ev_datetime = preferred_origin.time

    # Get the max radius
    mags = mw_rrup_data[:, 0]
    rrups = mw_rrup_data[:, 1]

    if preferred_magnitude <= mags.min():
        rrup = rrups.min()
    elif preferred_magnitude >= mags.max():
        rrup = rrups.max()
    else:
        interpolator = interp1d(mags, rrups, kind="cubic")
        rrup = float(interpolator(preferred_magnitude))
    maxradius = obspy.geodetics.kilometers2degrees(rrup)

    inv_sub = inventory.select(
        latitude=event_lat,
        longitude=event_lon,
        maxradius=maxradius,
        starttime=ev_datetime,
        endtime=ev_datetime,
    )

    return inv_sub


def fetch_sta_extraction(
    station: Station,
    network: Network,
    event_cat: Event,
    event_id: str,
    pref_mag: float,
    pref_mag_type: str,
    site_table: pd.DataFrame,
):
    """
    Fetch the extraction table for a station to be added to the station extraction table
    such as the vs30, z1p0, r_hyp, ds_mean, ds_std, and ptime_est.

    Parameters
    ----------
    station : Station
        The station to fetch the data for
    network : Network
        The network of the station
    event_cat : Event
        The event catalogue to fetch the data from
    event_id : str
        The event id
    pref_mag : float
        The preferred magnitude
    pref_mag_type : str
        The preferred magnitude type
    site_table : pd.DataFrame
        The site table to extract the vs30 value from

    Returns
    -------
    pd.DataFrame
        The station extraction table with the data for the station
    pd.DataFrame
        The skipped reasons if the station has an issue getting a phase arrival
    """
    config = cfg.Config()

    # Get the preferred_origin
    preferred_origin = event_cat.preferred_origin()
    ev_lat = preferred_origin.latitude
    ev_lon = preferred_origin.longitude

    # Get the r_hyp
    dist, _, _ = obspy.geodetics.gps2dist_azimuth(
        ev_lat,
        ev_lon,
        station.latitude,
        station.longitude,
    )
    r_epi = dist / 1000
    ev_depth = preferred_origin.depth / 1000
    r_hyp = ((r_epi) ** 2 + (ev_depth + station.elevation / 1000) ** 2) ** 0.5

    # Get the vs30 / z1p0 value
    site_row = site_table.loc[
        (site_table["net"] == network.code) & (site_table["sta"] == station.code), :
    ]
    vs30 = config.get_value("vs30") if site_row.empty else site_row["Vs30"].values[0]
    z1p0 = (
        estimations.chiou_young_08_calc_z1p0(vs30)
        if site_row.empty
        else site_row["Z1.0"].values[0] / 1000
    )

    rake = 90  # Assume strike-slip
    # Predict significant duration time from Afshari and Stewart (2016)
    input_df = pd.DataFrame(
        {
            "mag": [pref_mag],
            "rake": [rake],
            "rrup": [r_hyp],
            "vs30": [vs30],
            "z1pt0": [z1p0],
        }
    )
    result_df = wrapper.run_gmm(
        constants.GMM.AS_16,
        constants.TectType.ACTIVE_SHALLOW,
        input_df,
        "Ds595",
    )
    ds_mean = result_df["Ds595_mean"].values[0]
    ds_std = result_df["Ds595_std_Total"].values[0]

    deg = kilometers2degrees(r_epi)

    model = TauPyModel(model="iasp91")

    # Esitimate the P-wave travel time
    # Loop through the priority phase list to find the first available P-wave arrival
    priority_phase_list = config.get_value("priority_phase_list")
    p_arrivals = []
    for phase_list in priority_phase_list:
        p_arrivals = model.get_travel_times(
            source_depth_in_km=preferred_origin.depth / 1000,
            distance_in_degree=deg,
            phase_list=phase_list,
        )
        if p_arrivals:
            break

    if p_arrivals:
        ptime_est = preferred_origin.time + p_arrivals[0].time

        # Create the station_extraction_table
        station_extraction_table = pd.DataFrame(
            {
                "provider": ["GEONET"],
                "net": [network.code],
                "sta": [station.code],
                "evid": [event_id],
                "mag": [pref_mag],
                "pref_mag_type": [pref_mag_type],
                "r_hyp": [r_hyp],
                "vs30": [vs30],
                "z1p0": [z1p0],
                "ds_mean": [ds_mean],
                "ds_std": [ds_std],
                "phase": [p_arrivals[0].name],
                "ptime_est": [ptime_est],
            }
        )

        return station_extraction_table, pd.DataFrame()
    else:
        print(
            f"No phase arrivals found for {network.code}.{station.code} at event {event_id}"
        )
        skipped_reason = pd.DataFrame(
            {
                "record_id": [f"{event_id}_{station.code}"],
                "skipped_reason": ["No P-wave arrival estimate found"],
            }
        )
        return pd.DataFrame(), skipped_reason


def fetch_event_data(
    event_id: str,
    event_cat: Event,
    inventory: Inventory,
    site_table: pd.DataFrame,
    mw_rrup_data: np.ndarray,
    only_sites: list[str] = None,
    only_record_ids: pd.DataFrame = None,
    n_procs: int = 1,
):
    """
    Fetch the event data from the geonet client to form the event and station_extraction dataframes

    Parameters
    ----------
    event_id : str
        The event id to fetch the data for
    event_cat : Event
        The event catalogue to fetch the data from
    inventory : Inventory
        The inventory of the stations from all networks to extract the stations from
    site_table : pd.DataFrame
        The site table to extract the vs30 value from
    mw_rrup_data : np.ndarray
        The Mw_rrup data to get the interpolation function to determine the max radius.
    only_sites : list[str] (optional)
        Will only fetch the data for the sites in the list
    only_record_ids : pd.DataFrame (optional)
        Will only fetch the data for the record ids in the df, should all be a subset of the only_sites list
    n_procs : int (optional)
        The number of processes to run, to multiprocess over sites (when not using mp over events)

    Returns
    -------
    EventData
        The parsed event data.
    """
    # Get the event line
    event_line = fetch_event_line(event_cat, event_id)
    station_extraction_table = pd.DataFrame()
    skipped_records = pd.DataFrame()

    if event_line is not None:

        # Get Networks / Stations within a certain radius of the event
        inv_sub_sta = get_stations_within_radius(event_cat, mw_rrup_data, inventory)

        # Create a filtered list of tuples (network, station)
        filtered_stations = [
            (network, station)
            for network in inv_sub_sta
            for station in network
            if only_sites is None or station.code in only_sites
        ]

        # Further filter the list based on only_record_ids if defined
        if only_record_ids is not None:
            event_only_record_ids = only_record_ids[
                only_record_ids["record_id"].str.contains(f"^{event_id}_")
            ]
            if event_only_record_ids.empty:
                return [], [], [], []
            filtered_stations = [
                (network, station)
                for network, station in filtered_stations
                if not event_only_record_ids[
                    event_only_record_ids["record_id"].str.contains(f"_{station.code}_")
                ].empty
            ]

        if n_procs > 1:
            with mp.Pool(n_procs) as pool:
                results = pool.starmap(
                    functools.partial(
                        fetch_sta_extraction,
                        event_cat=event_cat,
                        event_id=event_id,
                        pref_mag=event_line[7],
                        pref_mag_type=event_line[8],
                        site_table=site_table,
                    ),
                    [(station, network) for network, station in filtered_stations],
                )
        else:
            results = [
                fetch_sta_extraction(
                    station,
                    network,
                    event_cat,
                    event_id,
                    event_line[7],
                    event_line[8],
                    site_table,
                )
                for network, station in filtered_stations
            ]

        # Concat the results for the station extraction table
        if results:
            station_extraction_results, skipped_records = zip(*results)
            station_extraction_table = pd.concat(
                station_extraction_results, ignore_index=True
            )
            skipped_records = pd.concat(skipped_records, ignore_index=True)

    return EventData(event_line, station_extraction_table, skipped_records)


def process_batch(
    batch_events: np.ndarray[str],
    batch_index: int,
    main_dir: Path,
    client_NZ: FDSN_Client,
    inventory: Inventory,
    site_table: pd.DataFrame,
    mw_rrup_data: np.ndarray,
    n_procs: int = 1,
    only_sites: list[str] = None,
    only_record_ids: pd.DataFrame = None,
    mp_sites: bool = False,
):
    """
    Process a batch of events to fetch the event data and create the dataframes

    Parameters
    ----------
    batch_events : np.ndarray[str]
        The array of event ids to fetch the data for
    batch_index : int
        The index of the current batch
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    client_NZ : FDSN_Client
        The geonet client to fetch the data from New Zealand
    inventory : Inventory
        The inventory of the stations from all networks to extract the stations from
    site_table : pd.DataFrame
        The site table to extract the vs30 value from
    mw_rrup_data : np.ndarray
        The Mw_rrup data to get the interpolation function to determine the max radius.
    n_procs : int (optional)
        The number of processes to run
    only_sites : list[str] (optional)
        Will only fetch the data for the sites in the list
    only_record_ids : pd.DataFrame (optional)
        Will only fetch the data for the record ids in the df, should all be a subset of the only_sites list
    mp_sites : bool (optional)
        Whether to multiprocess over sites (when not using mp over events)
    """
    # Get the catalogue information
    catalog_dict = {
        event_id: client_NZ.get_events(eventid=event_id)[0] for event_id in batch_events
    }

    # Fetch results
    if mp_sites:
        results = [
            fetch_event_data(
                event_id,
                catalog_dict[event_id],
                inventory,
                site_table,
                mw_rrup_data,
                only_sites,
                only_record_ids,
                n_procs,
            )
            for event_id in batch_events
        ]
    else:
        with mp.Pool(n_procs) as p:
            results = p.starmap(
                functools.partial(
                    fetch_event_data,
                    inventory=inventory,
                    site_table=site_table,
                    mw_rrup_data=mw_rrup_data,
                    only_sites=only_sites,
                    only_record_ids=only_record_ids,
                    n_procs=1,
                ),
                [(event_id, catalog_dict[event_id]) for event_id in batch_events],
            )

    # Extract the results
    event_data, station_extraction_data, skipped_records = [], [], []
    for result in results:
        (
            finished_event_data,
            finished_sta_extraction_data,
            finished_skipped_records,
        ) = result
        if finished_event_data is not None:
            event_data.append(finished_event_data)
            station_extraction_data.append(finished_sta_extraction_data)
            skipped_records.append(finished_skipped_records)

    # Create the output directory for the batch files
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    batch_dir = flatfile_dir / "geonet_batch_files"

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
    # Save the dataframes with a suffix
    event_df.to_csv(
        batch_dir / f"earthquake_source_table_{batch_index}.csv", index=False
    )

    # Combine the station extraction dataframes
    if station_extraction_data:
        station_extraction_df = pd.concat(station_extraction_data, ignore_index=True)
        # Save the station extraction table
        station_extraction_df.to_csv(
            batch_dir / f"station_extraction_table_{batch_index}.csv", index=False
        )

    if skipped_records:
        # Combine the skipped records dataframes
        skipped_records_df = pd.concat(skipped_records, ignore_index=True)
        # Save the skipped records
        skipped_records_df.to_csv(
            batch_dir / f"skipped_records_{batch_index}.csv", index=False
        )


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

    Returns
    -------
    pd.DataFrame
        The dataframe with the earthquake data from the geonet website
    """
    # Define bbox for New Zealand
    config = cfg.Config()
    bbox = ",".join([str(coord) for coord in config.get_value("bbox")])

    # Send API request for the date ranges required
    geonet_url = config.get_value("geonet_url")
    endpoint = (
        f"{geonet_url}/count?bbox={bbox}&startdate={start_date}&enddate={end_date}"
    )
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
    config = cfg.Config()
    min_mag = config.get_value("min_mag")
    max_mag = config.get_value("max_mag")

    # Loop over the dates to extract the csv data to a dataframe
    dfs = []
    for index, first_date in enumerate(response_dates[1:]):
        second_date = response_dates[index]
        endpoint = (
            f"{geonet_url}/csv?bbox={bbox}&startdate={first_date}&enddate={second_date}"
        )
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
    batch_size: int = 500,
    only_event_ids: list[str] = None,
    only_sites: list[str] = None,
    only_record_ids_ffp: Path = None,
    real_time: bool = False,
    mp_sites: bool = False,
    add_tmp_arrays: bool = False,
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
    batch_size : int (optional)
        The size of the batches to run, default is 500
    only_event_ids : list[str] (optional)
        Will only fetch the data for the event ids in the list (Must be in the start and end date range)
    only_sites : list[str] (optional)
        Will only fetch the data for the sites in the list
    only_record_ids_ffp : Path (optional)
        Will only fetch the data for the record ids in the df, will override the only_sites, only_event_ids lists
    real_time : bool (optional)
        If the function is being used in real time use a different client, default is False
    mp_sites : bool (optional)
        Whether to multiprocess over sites (when not using mp over events)
    add_tmp_arrays : bool (optional)
        Whether to add temporary array stations to the inventory, default is False
    """
    if only_record_ids_ffp:
        # Read the only record ids file
        only_record_ids = pd.read_csv(only_record_ids_ffp)
        # extract the event ids from the record ids
        event_ids = list(
            {record_id.split("_")[0] for record_id in only_record_ids["record_id"]}
        )
        # extract the sites from the record ids to replace the only_sites list
        only_sites = list(
            {record_id.split("_")[1] for record_id in only_record_ids["record_id"]}
        )
    else:
        if not only_event_ids:
            # Get the earthquake data
            geonet = download_earthquake_data(start_date, end_date)

            # Get all event ids
            event_ids = geonet.publicid.unique().astype(str)
        else:
            event_ids = only_event_ids
        only_record_ids = None

    config = cfg.Config()
    if real_time:
        inventory = inventory_xml.get_provider_inventory(
            real_time=True, level="station"
        )
        client_NZ = FDSN_Client(base_url=config.get_value("real_time_url"))
    else:
        inventory = inventory_xml.get_full_inventory(
            add_tmp_arrays=add_tmp_arrays, level="station"
        )
        client_NZ = FDSN_Client("GEONET")

    # Get the rrup data
    mw_rrup_data = np.loadtxt(NZGMDB_DATA.fetch("Mw_rrup.txt"))

    # Get the site table
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    site_table = pd.read_csv(flatfile_dir / file_structure.PreFlatfileNames.SITE_TABLE)

    batch_dir = flatfile_dir / "geonet_batch_files"
    batch_dir.mkdir(exist_ok=True, parents=True)

    # Find files that have already been processed and get the suffix indexes and remove them from the event_ids
    processed_files = [f for f in batch_dir.iterdir() if f.is_file()]
    processed_suffixes = set(int(f.stem.split("_")[-1]) for f in processed_files)

    # Create batches from the event_ids
    batches = np.array_split(event_ids, np.ceil(len(event_ids) / batch_size))

    for index, batch in enumerate(batches):
        if index not in processed_suffixes:
            print(f"Processing batch {index + 1}/{len(batches)}")
            process_batch(
                batch,
                index,
                main_dir,
                client_NZ,
                inventory,
                site_table,
                mw_rrup_data,
                n_procs,
                only_sites,
                only_record_ids,
                mp_sites,
            )

    # Combine all the event and station_extraction dataframes
    event_dfs = []
    station_extraction_dfs = []
    skipped_records_dfs = []

    for file in batch_dir.iterdir():
        if "earthquake_source_table" in file.stem:
            try:
                event_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")
        elif "station_extraction_table" in file.stem:
            try:
                station_extraction_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")
        elif "skipped_records" in file.stem:
            try:
                skipped_records_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")
                skipped_records_dfs.append(pd.DataFrame())

    if not station_extraction_dfs:
        raise custom_errors.NoStationsError(
            "No station magnitude data was found, please check the origin of the earthquake"
        )

    event_df = pd.concat(event_dfs, ignore_index=True)
    station_extraction_table = pd.concat(station_extraction_dfs, ignore_index=True)
    skipped_records = pd.concat(skipped_records_dfs, ignore_index=True)

    # Save the dataframes
    event_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_GEONET,
        index=False,
    )
    station_extraction_table.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.STATION_EXTRACTION_TABLE_GEONET,
        index=False,
    )
    skipped_records.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.GEONET_SKIPPED_RECORDS,
        index=False,
    )
