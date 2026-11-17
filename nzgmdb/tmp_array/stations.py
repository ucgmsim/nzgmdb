"""Find temporary array stations in NZ from FDSN providers and compare them with the pipeline config."""

import geopandas as gpd
import nzgeom.coastlines
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from shapely.geometry import Point

from nzgmdb.management import config as cfg
from nzgmdb.tmp_array.file_layout import STATION_COLUMNS

# Station services searched by default (event-only services such as USGS and EMSC are left out)
DEFAULT_PROVIDERS = [
    "AUSPASS",
    "BGR",
    "EIDA",
    "ETH",
    "GEOFON",
    "GEONET",
    "ICGC",
    "IESDMC",
    "INGV",
    "IPGP",
    "IRIS",
    "IRISPH5",
    "KNMI",
    "KOERI",
    "LMU",
    "NCEDC",
    "NIEP",
    "NOA",
    "ODC",
    "RASPISHAKE",
    "RESIF",
    "RESIFPH5",
    "SCEDC",
    "TEXNET",
    "UIB-NORSAR",
    "USP",
]
DEFAULT_CHANNELS = ["HH", "BH", "HN", "BN"]
# NZ and IU come from GeoNet in the main pipeline, SY is IRIS synthetic data
DEFAULT_EXCLUDED_NETWORKS = ["NZ", "IU", "SY"]


def open_channel_ids(
    client: FDSN_Client, query: dict
) -> set[tuple[str, str, str, str]] | None:
    """
    Get the channels a provider lists when restricted data is excluded.

    Parameters
    ----------
    client : FDSN_Client
        Provider client
    query : dict
        get_stations keyword arguments used for the full query

    Returns
    -------
    set[tuple[str, str, str, str]] | None
        (net, sta, loc, cha) ids of open channels,
        or None if the provider does not support the includerestricted option
    """
    try:
        inventory = client.get_stations(**query, includerestricted=False)
    except FDSNNoDataException:
        return set()
    except (FDSNException, ValueError, OSError):
        return None
    return {
        (network.code, station.code, channel.location_code, channel.code)
        for network in inventory
        for station in network
        for channel in station
    }


def query_provider(
    provider: str,
    start: UTCDateTime,
    end: UTCDateTime,
    bbox: tuple[float, float, float, float],
) -> list[list]:
    """
    Get one row per channel from a single provider, flagging restricted (embargoed) channels.

    Parameters
    ----------
    provider : str
        ObsPy FDSN provider name
    start : UTCDateTime
        Only stations operating after this time
    end : UTCDateTime
        Only stations operating before this time
    bbox : tuple[float, float, float, float]
        (min_lat, max_lat, min_lon, max_lon), min_lon > max_lon wraps across 180

    Returns
    -------
    list[list]
        Rows in STATION_COLUMNS order, restricted is "True", "False" or "" (unknown)
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    client = FDSN_Client(provider)
    query = dict(
        starttime=start,
        endtime=end,
        minlatitude=min_lat,
        maxlatitude=max_lat,
        minlongitude=min_lon,
        maxlongitude=max_lon,
        level="channel",
    )
    inventory = client.get_stations(**query)
    open_ids = open_channel_ids(client, query)

    rows = []
    for network in inventory:
        for station in network:
            for channel in station:
                # Restricted if marked closed, or hidden when restricted data is excluded
                status = (
                    channel.restricted_status
                    or station.restricted_status
                    or network.restricted_status
                )
                channel_id = (
                    network.code,
                    station.code,
                    channel.location_code,
                    channel.code,
                )
                if status == "closed" or (
                    open_ids is not None and channel_id not in open_ids
                ):
                    restricted = "True"
                elif status == "open" or open_ids is not None:
                    restricted = "False"
                else:
                    restricted = ""
                rows.append(
                    [
                        provider,
                        network.code,
                        station.code,
                        station.latitude,
                        station.longitude,
                        station.elevation,
                        channel.code[:2],
                        channel.location_code,
                        channel.start_date,
                        channel.end_date,
                        restricted,
                    ]
                )
    return rows


def _combine_restricted(values: pd.Series) -> str:
    """
    Combine channel restricted flags into one flag for a station row.

    Parameters
    ----------
    values : pd.Series
        Restricted flags ("True", "False" or "")

    Returns
    -------
    str
        "True" if any channel is restricted, "" if any is unknown, otherwise "False"
    """
    if (values == "True").any():
        return "True"
    return "" if (values == "").any() else "False"


def get_station_table(
    start: UTCDateTime,
    end: UTCDateTime,
    providers: list[str],
    channels: list[str],
    excluded_networks: list[str],
    bbox: tuple[float, float, float, float],
    coastline_filter: bool = True,
) -> pd.DataFrame:
    """
    Build the station request table from all providers.

    Providers that fail are skipped and reported, so one bad service does not stop the run.

    Parameters
    ----------
    start : UTCDateTime
        Only stations operating after this time
    end : UTCDateTime
        Only stations operating before this time
    providers : list[str]
        ObsPy FDSN provider names, earlier providers win for duplicate stations
    channels : list[str]
        2 character channel prefixes to keep e.g. HH
    excluded_networks : list[str]
        Network codes to drop
    bbox : tuple[float, float, float, float]
        (min_lat, max_lat, min_lon, max_lon)
    coastline_filter : bool, optional
        Keep only stations on the NZ mainland coastline polygons, by default True

    Returns
    -------
    pd.DataFrame
        One row per provider / net / sta / chan prefix / loc, with the widest
        start_date and end_date ("" when still running)
    """
    rows = []
    for provider in providers:
        try:
            provider_rows = query_provider(provider, start, end, bbox)
        except (FDSNException, ValueError, OSError) as e:
            print(f"Skipping provider {provider}: {type(e).__name__}: {e}")
            continue
        print(f"{provider}: {len(provider_rows)} channels")
        rows.extend(provider_rows)

    station_df = pd.DataFrame(rows, columns=STATION_COLUMNS)
    station_df = station_df[
        station_df["chan"].isin(channels) & ~station_df["net"].isin(excluded_networks)
    ]
    if station_df.empty:
        return station_df

    if coastline_filter:
        nz_coast = nzgeom.coastlines.get_NZ_coastlines().to_crs("EPSG:4326").union_all()
        points = gpd.GeoSeries(
            [Point(lon, lat) for lon, lat in zip(station_df["lon"], station_df["lat"])]
        )
        station_df = station_df[points.within(nz_coast).to_numpy()]

    # A missing end date means still running, keep it open ("") rather than the latest epoch end
    station_df["start_date"] = station_df["start_date"].map(str)
    station_df["open"] = station_df["end_date"].isna()
    station_df["end_date"] = station_df["end_date"].map(
        lambda x: "" if x is None else str(x)
    )
    key = ["net", "sta", "chan", "loc"]
    grouped = station_df.groupby(key + ["provider"], sort=False)
    station_df = grouped.agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        elev=("elev", "first"),
        start_date=("start_date", "min"),
        end_date=("end_date", "max"),
        open=("open", "any"),
        restricted=("restricted", _combine_restricted),
    ).reset_index()
    station_df.loc[station_df["open"], "end_date"] = ""

    # The same station can be served by several providers, keep the first in provider order
    station_df["provider_order"] = station_df["provider"].map(providers.index)
    station_df = station_df.sort_values(["provider_order"] + key).drop_duplicates(key)
    return station_df[STATION_COLUMNS].reset_index(drop=True)


def find_networks_missing_from_config(station_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find networks in the station table that the pipeline config does not know about.

    The pipeline only fetches inventories for networks listed in config.yaml
    (main_providers_networks and tmp_array_providers_networks).

    Parameters
    ----------
    station_df : pd.DataFrame
        Station table from get_station_table

    Returns
    -------
    pd.DataFrame
        One row per missing provider / network with station count and date range
    """
    config = cfg.Config()
    known_networks = {
        net
        for key in ("main_providers_networks", "tmp_array_providers_networks")
        for nets in config.get_value(key).values()
        for net in nets
    }
    missing_df = station_df[~station_df["net"].isin(known_networks)]
    return (
        missing_df.groupby(["provider", "net"])
        .agg(
            n_stations=("sta", "nunique"),
            first_start=("start_date", "min"),
            last_end=("end_date", "max"),
        )
        .reset_index()
    )
