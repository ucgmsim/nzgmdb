"""
This file contains the functions needed to convert a set of mseed files to the gmprocess format / file structure.
This currently handles both the old style of data storage and the new style of data.
"""

import datetime
import functools
import multiprocessing
from json import JSONDecodeError
from pathlib import Path

import requests
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core import Stream

from nzgmdb.management import config as cfg
from nzgmdb.mseed_management import creation, reading


def gen_station_xml(station: str, client: FDSN_Client, output_dir: Path):
    """
    Generates the station xml file for a given station

    Parameters
    ----------
    station : str
        The station code
    client : FDSN_Client
        The client to get the station information
    output_dir : Path
        The directory to save the station xml file

    Returns
    -------
    bool
        Whether the station xml file was generated successfully
    """
    config = cfg.Config()
    channel_codes = ",".join(config.get_value("channel_codes"))

    try:
        station_info = client.get_stations(
            station=station, channel=channel_codes, level="response"
        )
    except FDSNNoDataException:
        print(f"No data for {station}")
        return False

    network = station_info.networks[0].code

    station_info.write(output_dir / f"{network}.{station}.xml", format="STATIONXML")
    return True


def get_comcat_id(mseed: Stream, event_lat: float, event_lon: float, event_mag: float):
    """
    Get the comcat id from the mseed file by searching the USGS database with range on parameters
    such as latitude, longitude, magnitude, and time of the event

    Parameters
    ----------
    mseed : Stream
        The mseed file
    event_lat : float
        The latitude of the event
    event_lon : float
        The longitude of the event
    event_mag : float
        The magnitude of the event

    Returns
    -------
    str
        The comcat id of the event or None if not found
    """
    # Extract the config information
    config = cfg.Config()
    search_radius = config.get_value("search_radius")
    mag_range = config.get_value("mag_range")
    time_range = config.get_value("time_range")
    base = config.get_value("gmprocess_url")

    # Get the origin time of the event
    # (This is because obspy trim function does not update start and end times)
    origin = max([tr.stats.starttime for tr in mseed])

    # Prep the start and end times
    starttime = str(origin - datetime.timedelta(seconds=time_range)).replace(" ", "T")
    endtime = str(origin + datetime.timedelta(seconds=time_range)).replace(" ", "T")

    url = (
        f"{base}?format=geojson&starttime={starttime}&endtime={endtime}&latitude={event_lat:.5f}"
        f"&longitude={event_lon:.5f}&maxradiuskm={search_radius}&minmagnitude={event_mag - mag_range:.2f}"
        f"&maxmagnitude={event_mag + mag_range:.2f}"
    )

    # Send the request
    try:
        r = requests.get(url)
        json = r.json()
    except JSONDecodeError:
        return None

    # Check length and grab the first event found
    if json["metadata"]["count"] == 0:
        return None
    else:
        comcat_id = json["features"][0]["id"]

    return comcat_id


def split_mseed(mseed: Stream, output_dir: Path):
    """
    Split the mseed file into individual components
    Note: This only keeps the channels starting with B or H

    Parameters
    ----------
    mseed : Stream
        The mseed file
    output_dir : Path
        The directory to save the mseed files
    """
    for trace in mseed:
        if trace.stats.channel[0] in ["B", "H"]:
            starttime = trace.stats.starttime.strftime("%Y%m%dT%H%M%SZ")
            endtime = trace.stats.endtime.strftime("%Y%m%dT%H%M%SZ")
            filename = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}.{trace.stats.channel}__{starttime}__{endtime}.mseed"
            creation.write_stream_to_mseed(trace, output_dir / filename)


def run_for_single_event(
    event_dir: Path,
    client_NZ: FDSN_Client,
    output_dir: Path,
):
    """
    Run the conversion for a single event.
    Will generate the station xml files and split the mseed files
    and will abort fully if no comcat id is found
    or for a single station if the station xml file cannot be generated.

    Parameters
    ----------
    event_dir : Path
        The path to the event directory
    client_NZ : FDSN_Client
        The client to get the event information
    output_dir : Path
        The directory to save the new files in the new format
    """
    # Get the event ID information
    event_id = event_dir.name

    # Get the catalog information
    cat = client_NZ.get_events(eventid=event_id)
    event_cat = cat[0]

    preferred_magnitude = event_cat.preferred_magnitude().mag
    preferred_origin = event_cat.preferred_origin()
    event_lat = preferred_origin.latitude
    event_lon = preferred_origin.longitude

    mseeds = list(event_dir.glob("**/*.mseed"))

    # Loop over every mseed till a comcat id is found
    # This gives more of a chance to find a valid comcat id
    comcat_id = None
    for mseed_ffp in mseeds:
        mseed = reading.read_mseed_to_stream(mseed_ffp)
        found_id = get_comcat_id(mseed, event_lat, event_lon, preferred_magnitude)
        if found_id is not None:
            comcat_id = found_id
            break

    if comcat_id is None:
        print(f"No comcat id for {event_id}")
        return None

    # Create the new directory
    new_event_dir = output_dir / comcat_id / "raw"
    new_event_dir.mkdir(parents=True, exist_ok=True)

    # For each mseed file, generate the station xml file and split the mseed file
    for mseed_ffp in mseeds:
        mseed = reading.read_mseed_to_stream(mseed_ffp)

        # Check that there is at least one trace with a channel starting with B or H
        if any(trace.stats.channel[0] in ["B", "H"] for trace in mseed):
            generated = gen_station_xml(
                mseed[0].stats.station, client_NZ, new_event_dir
            )

            if not generated:
                return None

            split_mseed(mseed, new_event_dir)


def get_event_dirs(main_dir: Path):
    """
    Get the event directories

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)

    Returns
    -------
    list
        A list of the event directories
    """
    waveform_dir = main_dir / "waveforms"
    event_dirs = [
        event_dir
        for year_dir in waveform_dir.iterdir()
        for event_dir in year_dir.iterdir()
    ]

    return event_dirs


def convert_mseed_to_gmprocess(main_dir: Path, output_dir: Path, n_procs: int = 1):
    """
    Converts mseed data to gmprocess format and file structure.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    output_dir : Path
        The directory to save the gmprocessed data
    n_procs : int (optional)
        The number of processes to use for processing
    """
    # Get Station Information from geonet clients
    client_NZ = FDSN_Client("GEONET")

    # Get the event_dirs
    event_dirs = get_event_dirs(main_dir)

    # Do multiprocessing over each event
    with multiprocessing.Pool(processes=n_procs) as pool:
        pool.map(
            functools.partial(
                run_for_single_event,
                client_NZ=client_NZ,
                output_dir=output_dir,
            ),
            event_dirs,
        )
