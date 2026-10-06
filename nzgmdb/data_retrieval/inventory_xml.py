"""
Fetches inventory data from the obspy FDSN client and saves it as StationXML files.
"""

import datetime
from pathlib import Path

import pandas as pd
from obspy import Inventory
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNException

from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors, file_structure

STATION_INFO_COLUMNS = [
    "provider",
    "net",
    "sta",
    "lat",
    "lon",
    "elev",
    "creation_date",
    "end_date",
    "chan",
    "loc",
    "loc_elev",
    "start_time",
    "end_time",
]


def get_provider_inventory(
    provider: str | None = None,
    networks: list[str] | None = None,
    channel_codes: str | None = None,
    stations: str = "*",
    level: str = "response",
    starttime: str = "2000-01-01",
    endtime: str | None = None,
    real_time: bool = False,
) -> Inventory | None:
    """
    Fetch inventory from a single FDSN provider within the configured bounding box.

    Parameters
    ----------
    provider : str, optional
        FDSN provider name or base URL (e.g. "GEONET", "IRIS"), required if `real_time` is False.
    networks : list[str], optional
        List of network codes to fetch. If None, fetches all networks.
    channel_codes : str, optional
        Channel codes filter. If None, uses the config default.
    stations : str, optional
        Station selector passed to FDSN, by default "*".
    level : str, optional
        StationXML detail level to request, by default "response".
    starttime : str, optional
        Start date (YYYY-MM-DD), by default "2000-01-01".
        Overridden to the last 14 days when `real_time` is True.
    endtime : str, optional
        End date (YYYY-MM-DD), by default today.
    real_time : bool, optional
        Whether to use the real-time FDSN url from the config, by default False.

    Returns
    -------
    Inventory or None
        The inventory for the provider, or None if no data was found.

    Raises
    ------
    ValueError
        If `provider` is None and `real_time` is False.
    """
    config = cfg.Config()
    channel_codes = (
        config.get_value("channel_codes") if channel_codes is None else channel_codes
    )
    endtime = datetime.date.today().isoformat() if endtime is None else endtime
    min_lon, min_lat, _, max_lat = config.get_value("bbox")
    # FDSN queries can't cross the antimeridian, no sites of interest lie past 180
    max_lon = 180

    if real_time:
        provider = config.get_value("real_time_url")
        # Only look at recent stations for real-time data to improve speed
        starttime = (datetime.date.today() - datetime.timedelta(days=14)).isoformat()
    elif provider is None:
        raise ValueError("Provider must be specified if not using real-time data.")

    try:
        # Client creation is inside the try as service discovery can fail if a provider is down
        client = FDSN_Client(provider)
        return client.get_stations(
            network="*" if networks is None else ",".join(networks),
            station=stations,
            channel=channel_codes,
            level=level,
            maxlatitude=max_lat,
            minlatitude=min_lat,
            maxlongitude=max_lon,
            minlongitude=min_lon,
            starttime=starttime,
            endtime=endtime,
        )
    except FDSNException as e:
        print(f"No inventory data found for provider {provider}: {type(e).__name__}")
        return None


def get_full_inventory(
    add_tmp_arrays: bool = False,
    level: str = "response",
    channel_codes: str | None = None,
    stations: str = "*",
    starttime: str = "2000-01-01",
    endtime: str | None = None,
    return_df: bool = False,
) -> Inventory | pd.DataFrame | None:
    """
    Fetch inventories from all configured providers and merge them.

    Parameters
    ----------
    add_tmp_arrays : bool, optional
        Whether to include temporary array providers, by default False.
    level : str, optional
        StationXML detail level to request, by default "response".
    channel_codes : str, optional
        Channel codes filter. If None, uses the config default.
    stations : str, optional
        Station selector passed to FDSN, by default "*".
    starttime : str, optional
        Start date (YYYY-MM-DD), by default "2000-01-01".
    endtime : str, optional
        End date (YYYY-MM-DD), by default today.
    return_df : bool, optional
        If True, return a DataFrame of station / channel info instead of the Inventory, by default False.

    Returns
    -------
    Inventory or pd.DataFrame or None
        The merged Inventory (None if no provider returned data),
        or a DataFrame with STATION_INFO_COLUMNS when `return_df` is True.
    """
    config = cfg.Config()
    # Copy so the shared config dictionary is not modified
    provider_networks = dict(config.get_value("main_providers_networks"))
    if add_tmp_arrays:
        provider_networks.update(config.get_value("tmp_array_providers_networks"))

    full_inventory = None
    station_info = []
    for provider, networks in provider_networks.items():
        inventory = get_provider_inventory(
            provider=provider,
            networks=networks,
            stations=stations,
            channel_codes=channel_codes,
            level=level,
            starttime=starttime,
            endtime=endtime,
        )
        if inventory is None:
            continue
        if full_inventory is None:
            full_inventory = inventory
        else:
            full_inventory += inventory

        if return_df:
            station_info.extend(
                [
                    provider,
                    network.code,
                    station.code,
                    station.latitude,
                    station.longitude,
                    station.elevation,
                    station.creation_date,
                    station.end_date,
                    channel.code[:2],
                    channel.location_code,
                    channel.depth,
                    channel.start_date,
                    channel.end_date,
                ]
                for network in inventory
                for station in network
                for channel in station.channels
            )

    if return_df:
        return (
            pd.DataFrame(station_info, columns=STATION_INFO_COLUMNS)
            .drop_duplicates(["provider", "net", "sta", "chan", "loc", "loc_elev"])
            .reset_index(drop=True)
        )

    return full_inventory


def fetch_and_save_inventory(
    main_dir: Path,
    stations: list[str],
    add_tmp_arrays: bool = False,
    starttime: str = "2000-01-01",
    endtime: str | None = None,
):
    """
    Fetch inventory data for the given stations and save each as a StationXML file.

    Parameters
    ----------
    main_dir : Path
        The main directory where the StationXML files will be saved.
    stations : list[str]
        A list of station codes to fetch the inventory data for.
    add_tmp_arrays : bool, optional
        Whether to include temporary array providers in the inventory fetch, by default False.
    starttime : str, optional
        The start time for the inventory data, by default "2000-01-01".
    endtime : str, optional
        The end time for the inventory data, by default today.
    """
    xml_dir = file_structure.get_stationxml_dir(main_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)

    inv = get_full_inventory(
        add_tmp_arrays=add_tmp_arrays,
        stations=",".join(stations),
        starttime=starttime,
        endtime=endtime,
    )
    if inv is None:
        raise custom_errors.InventoryNotFoundError("No inventory data found for the specified stations.")

    for sta in stations:
        sel = inv.select(station=sta)
        if not sel.networks:
            print(f"Warning: No inventory data found for station {sta}. Skipping.")
            continue
        sel.write(xml_dir / f"{sta}.xml", format="STATIONXML")
