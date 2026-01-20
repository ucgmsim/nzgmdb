"""
Fetches inventory data from the obspy FDSN client and saves it as StationXML files.
"""

import datetime
from pathlib import Path

from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNNoDataException

from nzgmdb.management import file_structure


def fetch_inventory(
    add_tmp_arrays: bool = False,
    level: str = "response",
    channel_codes: str | None = None,
    starttime: str = "2000-01-01",
    endtime: str = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d"),
):
    pass


def fetch_and_save_inventory(
    main_dir: Path,
    stations: list[str],
    starttime: str = "2000-01-01",
    endtime: str = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d"),
):
    """
    Fetches inventory data from the obspy FDSN client and saves it as StationXML files.

    Parameters
    ----------
    main_dir : Path
        The main directory where the StationXML files will be saved.
    stations : list[str]
        A list of station codes to fetch the inventory data for.
    starttime : str, optional
        The start time for the inventory data, by default "2000-01-01".
    endtime : str, optional
        The end time for the inventory data, by default the current date.
    """
    client = FDSN_Client("GEONET")

    xml_dir = file_structure.get_stationxml_dir(main_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)

    all_stations = ",".join(stations)

    try:
        inv = client.get_stations(
            network="NZ",
            station=all_stations,
            starttime=starttime,
            endtime=endtime,
            level="response",
        )
        for sta in stations:
            sel = inv.select(station=sta)
            if not sel.networks:
                print(f"Warning: No inventory data found for station {sta}. Skipping.")
                continue
            fname = xml_dir / f"NZ.{sta}.xml"
            sel.write(fname, format="STATIONXML")
    except FDSNNoDataException:
        print("No inventory data found for the specified stations and time range.")
