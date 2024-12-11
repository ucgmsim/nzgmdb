"""
Functions to manage Geonet Data
"""

import datetime
import io
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import requests
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.core.event import Event, Magnitude
from obspy.core.inventory import Inventory, Network, Station
from pandas.errors import EmptyDataError
from scipy.interpolate import interp1d

from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors, custom_multiprocess, file_structure
from nzgmdb.mseed_management import creation


def get_max_magnitude(magnitudes: list[Magnitude], mag_type: str):
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
    ev_ndef = (
        None
        if preferred_origin.quality.used_phase_count is None
        else preferred_origin.quality.used_phase_count
    )
    ev_nsta = (
        None
        if preferred_origin.quality.used_station_count is None
        else preferred_origin.quality.used_station_count
    )
    std = (
        None
        if preferred_origin.quality.standard_error is None
        else preferred_origin.quality.standard_error
    )

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
    mags: np.ndarray,
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
    mags : np.ndarray
        The magnitudes from the Mw_rrup file
    rrups : np.ndarray
        The rrups from the Mw_rrup file
    f_rrup : interp1d
        The cubic interpolation function for the magnitude distance relationship
    inventory : Inventory
        The inventory of the stations from all networks to extract the stations from

    Returns
    -------
    inv_sub : Inventory
        The subset of the inventory with the stations within the radius
    """
    preferred_magnitude = event_cat.preferred_magnitude().mag
    preferred_origin = event_cat.preferred_origin()
    event_lat = preferred_origin.latitude
    event_lon = preferred_origin.longitude

    # Get the max radius
    if preferred_magnitude < mags.min():
        rrup = np.array(rrups.min())
    elif preferred_magnitude > mags.max():
        rrup = np.array(rrups.max())
    else:
        rrup = f_rrup(preferred_magnitude)
    maxradius = obspy.geodetics.kilometers2degrees(rrup)

    inv_sub = inventory.select(
        latitude=event_lat, longitude=event_lon, maxradius=maxradius
    )

    return inv_sub


def fetch_sta_mag_line(
    station: Station,
    network: Network,
    event_cat: Event,
    event_id: str,
    main_dir: Path,
    client_NZ: FDSN_Client,
    pref_mag: float,
    pref_mag_type: str,
    site_table: pd.DataFrame,
):
    """
    Fetch the station magnitude line from the geonet client to be added to the sta_mag_df
    Also creates the mseed files for the station

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
    inventory : Inventory
        The inventory of the stations from all networks to extract the stations from
    pref_mag : float
        The preferred magnitude
    pref_mag_type : str
        The preferred magnitude type
    site_table : pd.DataFrame
        The site table to extract the vs30 value from
    mags : np.ndarray
        The magnitudes from the Mw_rrup file
    rrups : np.ndarray
        The rrups from the Mw_rrup file
    f_rrup : interp1d
        The cubic interpolation function for the magnitude distance relationship
    only_sites : list[str] (optional)
        Will only fetch the data for the sites in the list
    """
    sta_mag_line = []
    skipped_records = []
    # Get the clipping threshold
    config = cfg.Config()
    threshold = config.get_value("clip_threshold")

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
    r_hyp = ((r_epi) ** 2 + (ev_depth + station.elevation) ** 2) ** 0.5

    # Get the vs30 value
    site_vs30_row = site_table.loc[
        (site_table["net"] == network.code) & (site_table["sta"] == station.code),
        "Vs30",
    ]
    vs30 = None if site_vs30_row.empty else site_vs30_row.values[0]

    # Get the waveforms
    st = creation.get_waveforms(
        preferred_origin,
        client_NZ,
        network.code,
        station.code,
        event_cat.preferred_magnitude().mag,
        r_hyp,
        r_epi,
        vs30,
    )
    # Check that data was found
    if st is None:
        skipped_records.append([f"{event_id}_{station.code}", "No Waveform Data"])
        return sta_mag_line, skipped_records

    # Get the unique channels (Using first 2 keys) and locations
    unique_channels = set([(tr.stats.channel[:2], tr.stats.location) for tr in st])

    # Split the stream into mseeds
    mseeds = creation.split_stream_into_mseeds(st, unique_channels)

    # Get the station magnitudes
    station_magnitudes = [
        mag
        for mag in event_cat.station_magnitudes
        if mag.waveform_id.station_code == station.code
    ]

    for mseed in mseeds:
        # Check the data is not all 0's
        if all([np.allclose(tr.data, 0) for tr in mseed]):
            stats = mseed[0].stats
            skipped_records.append(
                [
                    f"{event_id}_{stats.station}_{stats.location}_{stats.channel}",
                    "All 0's",
                ]
            )
            continue

        # Create the directory structure for the given event
        year = event_cat.origins[0].time.year
        mseed_dir = file_structure.get_mseed_dir(main_dir, year, event_id)

        # Write the mseed file
        creation.write_mseed(mseed, event_id, station.code, mseed_dir)

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

    return sta_mag_line, skipped_records


def fetch_event_data(
    event_id: str,
    main_dir: Path,
    client_NZ: FDSN_Client,
    inventory: Inventory,
    site_table: pd.DataFrame,
    mags: np.ndarray,
    rrups: np.ndarray,
    f_rrup: interp1d,
    only_sites: list[str] = None,
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
    inventory : Inventory
        The inventory of the stations from all networks to extract the stations from
    site_table : pd.DataFrame
        The site table to extract the vs30 value from
    mags : np.ndarray
        The magnitudes from the Mw_rrup file
    rrups : np.ndarray
        The rrups from the Mw_rrup file
    f_rrup : interp1d
        The cubic interpolation function for the magnitude distance relationship
    only_sites : list[str] (optional)
        Will only fetch the data for the sites in the list
    """
    # Get the catalog information
    cat = client_NZ.get_events(eventid=event_id)
    event_cat = cat[0]

    # Get the event line
    event_line = fetch_event_line(event_cat, event_id)

    if event_line is not None:
        sta_mag_lines = []
        skipped_records = []

        # Get Networks / Stations within a certain radius of the event
        inv_sub_sta = get_stations_within_radius(
            event_cat, mags, rrups, f_rrup, inventory
        )

        # Loop through the Inventory Subset of Networks / Stations
        for network in inv_sub_sta:
            for station in network:
                # Check if the station is in the only_sites list if only_sites is defined
                if only_sites is None or station.code in only_sites:
                    # Get the station magnitude lines
                    sta_mag_line, new_skipped_records = fetch_sta_mag_line(
                        station,
                        network,
                        event_cat,
                        event_id,
                        main_dir,
                        client_NZ,
                        event_line[7],
                        event_line[8],
                        site_table,
                    )
                    sta_mag_lines.extend(sta_mag_line)
                    skipped_records.extend(new_skipped_records)
    else:
        sta_mag_lines, skipped_records = None, None

    return event_line, sta_mag_lines, skipped_records


def process_batch(
    batch_events: np.ndarray[str],
    batch_index: int,
    main_dir: Path,
    client_NZ: FDSN_Client,
    inventory: Inventory,
    site_table: pd.DataFrame,
    mags: np.ndarray,
    rrups: np.ndarray,
    f_rrup: interp1d,
    n_procs: int = 1,
    only_sites: list[str] = None,
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
    mags : np.ndarray
        The magnitudes from the Mw_rrup file
    rrups : np.ndarray
        The rrups from the Mw_rrup file
    f_rrup : interp1d
        The cubic interpolation function for the magnitude distance relationship
    n_procs : int (optional)
        The number of processes to run
    only_sites : list[str] (optional)
        Will only fetch the data for the sites in the list
    """
    # Use custom_multiprocess to fetch the event data
    results = custom_multiprocess.custom_multiprocess(
        fetch_event_data,
        batch_events,
        n_procs,
        main_dir,
        client_NZ,
        inventory,
        site_table,
        mags,
        rrups,
        f_rrup,
        only_sites,
    )

    # Extract the results
    event_data, sta_mag_data, skipped_records = [], [], []
    for result in results:
        finished_event_data, finished_sta_mag_data, finished_skipped_records = result
        event_data.append(finished_event_data)
        sta_mag_data.extend(finished_sta_mag_data)
        skipped_records.extend(finished_skipped_records)

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
        batch_dir / f"geonet_skipped_records_{batch_index}.csv", index=False
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
    real_time: bool = False,
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
    real_time : bool (optional)
        If the function is being used in real time use a different client, default is False
    """
    if not only_event_ids:
        # Get the earthquake data
        geonet = download_earthquake_data(start_date, end_date)

        # Get all event ids
        event_ids = geonet.publicid.unique().astype(str)
    else:
        event_ids = only_event_ids

    # Set constants
    config = cfg.Config()
    channel_codes = ",".join(config.get_value("channel_codes"))

    if real_time:
        client_NZ = FDSN_Client(base_url=config.get_value("real_time_url"))
    else:
        # Get Station Information from geonet clients
        client_NZ = FDSN_Client("GEONET")
    inventory = client_NZ.get_stations(channel=channel_codes, level="response")

    # Get the data_dir
    data_dir = file_structure.get_data_dir()

    mw_rrup = np.loadtxt(data_dir / "Mw_rrup.txt")
    mags = mw_rrup[:, 0]
    rrups = mw_rrup[:, 1]
    # Generate cubic interpolation for magnitude distance relationship
    f_rrup = interp1d(mags, rrups, kind="cubic")

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
                mags,
                rrups,
                f_rrup,
                n_procs,
                only_sites,
            )

    # Combine all the event and sta_mag dataframes
    event_dfs = []
    sta_mag_dfs = []
    skipped_records_dfs = []

    for file in batch_dir.iterdir():
        if "earthquake_source_table" in file.stem:
            try:
                event_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")
        elif "station_magnitude_table" in file.stem:
            try:
                sta_mag_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")
        elif "geonet_skipped_records" in file.stem:
            try:
                skipped_records_dfs.append(pd.read_csv(file))
            except EmptyDataError:
                print(f"Warning: {file} is empty or has no valid columns to parse.")

    if not sta_mag_dfs:
        raise custom_errors.NoStationsError(
            "No station magnitude data was found, please check the origin of the earthquake"
        )

    event_df = pd.concat(event_dfs, ignore_index=True)
    sta_mag_df = pd.concat(sta_mag_dfs, ignore_index=True)
    skipped_records_df = pd.concat(skipped_records_dfs, ignore_index=True)

    # Save the dataframes
    event_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_GEONET,
        index=False,
    )
    sta_mag_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.STATION_MAGNITUDE_TABLE_GEONET,
        index=False,
    )
    skipped_records_df.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.GEONET_SKIPPED_RECORDS,
        index=False,
    )
