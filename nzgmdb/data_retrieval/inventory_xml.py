"""
Fetches inventory data from the obspy FDSN client and saves it as StationXML files.
"""

import datetime
from pathlib import Path

import pandas as pd
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure


def get_provider_inventory(
    provider: str = None,
    networks: list[str] = None,
    channel_codes: str | None = None,
    stations: str = "*",
    level: str = "response",
    starttime: str = "2000-01-01",
    endtime: str = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d"),
    real_time: bool = False,
):
    """
    Fetch inventory from a specified FDSN provider within configured bounding box.

    Parameters
    ----------
    provider : str, optional
        FDSN provider base URL, required if `real_time` is False.
    networks : list[str], optional
        List of network codes to fetch. If None, fetches all networks.
    channel_codes : str or None, optional
        Channel codes filter. If None, uses config default.
    stations : str, optional
        Station selector passed to FDSN, by default "*".
    level : str, optional
        StationXML detail level to request, by default "response".
    starttime : str, optional
        Start date (YYYY-MM-DD), by default "2000-01-01".
    endtime : str, optional
        End date (YYYY-MM-DD), by default today.
    real_time : bool, optional
        Whether to use real-time data source from config, by default False.
    """
    config = cfg.Config()
    channel_codes = (
        config.get_value("channel_codes") if channel_codes is None else channel_codes
    )
    bbox = config.get_value("bbox")  # [min_lon, min_lat, max_lon, max_lat]
    min_lon, min_lat, max_lon, max_lat = bbox
    max_lon = 180  # Due to issues with FDSN of passing barrier (no land past this point for sites that are of interest)
    if real_time:
        client = FDSN_Client(base_url=config.get_value("real_time_url"))
        # Adjust the start time to be more recent for real-time data to improve speed
        starttime = datetime.datetime.strftime(
            datetime.datetime.now() - datetime.timedelta(days=14), "%Y-%m-%d"
        )
    else:
        if provider is None:
            raise ValueError("Provider must be specified if not using real-time data.")
        client = FDSN_Client(provider)
    networks = "*" if networks is None else ",".join(networks)
    try:
        inv = client.get_stations(
            network=networks,
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
    except FDSNException:
        print(
            f"No inventory data found for provider {provider} with the specified parameters."
        )
        inv = None
    return inv


def get_full_inventory(
    add_tmp_arrays: bool = False,
    level: str = "response",
    channel_codes: str | None = None,
    stations: str = "*",
    starttime: str = "2000-01-01",
    endtime: str = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d"),
    return_df: bool = False,
):
    """
    Fetch inventories from all configured providers and optionally return station/channel metadata.

    Parameters
    ----------
    add_tmp_arrays : bool, optional
        Whether to include temporary array providers, by default False.
    level : str, optional
        StationXML detail level to request, by default "response".
    channel_codes : str or None, optional
        Channel codes filter. If None, uses config default.
    stations : str, optional
        Station selector passed to FDSN, by default "*".
    starttime : str, optional
        Start date (YYYY-MM-DD), by default "2000-01-01".
    endtime : str, optional
        End date (YYYY-MM-DD), by default today.
    return_df : bool, optional
        If True, return a DataFrame of station/channel info; otherwise return merged Inventory.

    Returns
    -------
    pandas.DataFrame or obspy.core.inventory.inventory.Inventory
        DataFrame when `return_df=True`, else the merged ObsPy Inventory.
    """
    config = cfg.Config()
    provider_networks = config.get_value("main_providers_networks")
    if add_tmp_arrays:
        provider_networks.update(config.get_value("tmp_array_providers_networks"))
    return_inv = None
    info_dfs = []
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
        if return_inv is None:
            return_inv = inventory
        else:
            return_inv += inventory

        if return_df:
            station_info = [
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
            ]

            info_dfs.append(
                pd.DataFrame(
                    station_info,
                    columns=[
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
                    ],
                )
            )

    if return_df:
        if not info_dfs:
            return pd.DataFrame(
                columns=[
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
            )

        all_info_df = pd.concat(info_dfs, ignore_index=True)
        return all_info_df.drop_duplicates(
            ["provider", "net", "sta", "chan", "loc", "loc_elev"]
        ).reset_index(drop=True)

    return return_inv


def fetch_and_save_inventory(
    main_dir: Path,
    stations: list[str],
    add_tmp_arrays: bool = False,
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
    add_tmp_arrays : bool, optional
        Whether to include temporary array providers in the inventory fetch, by default False.
    starttime : str, optional
        The start time for the inventory data, by default "2000-01-01".
    endtime : str, optional
        The end time for the inventory data, by default the current date.
    """
    xml_dir = file_structure.get_stationxml_dir(main_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)

    all_stations = ",".join(stations)

    try:
        inv = get_full_inventory(
            add_tmp_arrays=add_tmp_arrays,
            stations=all_stations,
            starttime=starttime,
            endtime=endtime,
        )
        for sta in stations:
            sel = inv.select(station=sta)
            if not sel.networks:
                print(f"Warning: No inventory data found for station {sta}. Skipping.")
                continue
            fname = xml_dir / f"{sta}.xml"
            sel.write(fname, format="STATIONXML")
    except FDSNNoDataException:
        print("No inventory data found for the specified stations and time range.")


df = pd.read_csv(
    "/media/joel/data/nzgmdb/tmp_arrays/rch_run_template/flatfiles/station_table_all.csv"
)
df_chan = df[df["chan"].isin(["EH", "DH"])]
unique_sites = df_chan["sta"].unique()
main_dir = Path(
    "/media/joel/data/nzgmdb/tmp_arrays/rch_run_template/flatfiles/dh_eh_xmls"
)
fetch_and_save_inventory(main_dir, unique_sites, add_tmp_arrays=True)
