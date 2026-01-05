"""
Fetches inventory data from the obspy FDSN client and saves it as StationXML files.
"""

import datetime
from pathlib import Path

from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.management import file_structure


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

    for sta in stations:
        inv = client.get_stations(
            network="NZ",
            station=sta,
            starttime=starttime,
            endtime=endtime,
            level="response",
        )
        fname = xml_dir / f"NZ.{sta}.xml"
        inv.write(fname, format="STATIONXML")
