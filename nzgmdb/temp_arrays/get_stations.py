"""Fetch station/channel metadata from FDSN providers and write to CSV."""

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import nzgeom.coastlines
import pandas as pd
import typer
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSN_Client
from shapely.geometry import Point

app = typer.Typer(pretty_exceptions_enable=False)

URL_MAPPINGS = {
    "AUSPASS": "http://auspass.edu.au",
    "BGR": "http://eida.bgr.de",
    "EIDA": "http://eida-federator.ethz.ch",
    "ETH": "http://eida.ethz.ch",
    "EMSC": "http://www.seismicportal.eu",
    "GEONET": "http://service.geonet.org.nz",
    "GEOFON": "http://geofon.gfz-potsdam.de",
    "GFZ": "http://geofon.gfz-potsdam.de",
    "ICGC": "http://ws.icgc.cat",
    "IESDMC": "http://batsws.earth.sinica.edu.tw",
    "INGV": "http://webservices.ingv.it",
    "IPGP": "http://ws.ipgp.fr",
    "IRIS": "http://service.iris.edu",
    "IRISPH5": "http://service.iris.edu",
    "ISC": "http://www.isc.ac.uk",
    "KNMI": "http://rdsa.knmi.nl",
    "KOERI": "http://eida.koeri.boun.edu.tr",
    "LMU": "https://erde.geophysik.uni-muenchen.de",
    "NCEDC": "https://service.ncedc.org",
    "NIEP": "http://eida-sc3.infp.ro",
    "NOA": "http://eida.gein.noa.gr",
    "ODC": "http://www.orfeus-eu.org",
    "ORFEUS": "http://www.orfeus-eu.org",
    "RESIF": "http://ws.resif.fr",
    "RESIFPH5": "http://ph5ws.resif.fr",
    "RASPISHAKE": "https://data.raspberryshake.org",
    "SCEDC": "http://service.scedc.caltech.edu",
    "TEXNET": "http://rtserve.beg.utexas.edu",
    "UIB-NORSAR": "http://eida.geo.uib.no",
    "USGS": "http://earthquake.usgs.gov",
    "USP": "http://sismo.iag.usp.br",
}


def is_point_inside_nz(lat: float, lon: float, *, nz_coast: gpd.GeoDataFrame) -> bool:
    """Return True if the given point lies within NZ coastline polygons.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    nz_coast : geopandas.GeoDataFrame
        Coastline polygons in EPSG:4326.

    Returns
    -------
    bool
        True if the point is contained in any polygon.
    """

    p = Point(lon, lat)

    # Iterate geometries directly so we don't rely on pandas/geopandas methods here.
    return any(getattr(g, "contains")(p) for g in nz_coast.geometry)


def _parse_time(value: str) -> UTCDateTime:
    """Parse a time value into ``UTCDateTime``.

    Parameters
    ----------
    value : str
        Input time string parseable by ObsPy. If "now" (case-insensitive), the
        current time is used.

    Returns
    -------
    UTCDateTime
        Parsed time.

    Raises
    ------
    ValueError
        If the value cannot be parsed.
    """
    if value.strip().lower() == "now":
        return UTCDateTime()
    return UTCDateTime(value)


def _iter_providers(providers: list[str] | None) -> list[str]:
    """Build the provider list for querying.

    Parameters
    ----------
    providers : list[str] | None
        Optional list of provider names. If None or empty, uses all known
        providers.

    Returns
    -------
    list[str]
        Provider names.
    """
    if not providers:
        return sorted(URL_MAPPINGS)
    return providers


def collect_station_rows(
    providers: list[str],
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    minlatitude: float,
    maxlatitude: float,
    minlongitude: float,
    maxlongitude: float,
    coastline_filter: bool,
) -> list[list[object]]:
    """Collect station/channel rows from configured providers.

    Parameters
    ----------
    providers : list[str]
        Provider names.
    starttime : UTCDateTime
        Metadata start time.
    endtime : UTCDateTime
        Metadata end time.
    minlatitude, maxlatitude : float
        Latitude bounding box.
    minlongitude, maxlongitude : float
        Longitude bounding box.
    coastline_filter : bool
        If True, filter stations to those within NZ coastline polygons.

    Returns
    -------
    list[list[object]]
        Rows matching the output CSV schema.
    """
    nz_coast = (
        nzgeom.coastlines.get_NZ_coastlines().to_crs("EPSG:4326")
        if coastline_filter
        else None
    )

    all_station_info: list[list[object]] = []

    for provider in providers:
        client = FDSN_Client(base_url=provider)
        networks = client.get_stations(
            starttime=starttime,
            endtime=endtime,
            minlatitude=minlatitude,
            maxlatitude=maxlatitude,
            minlongitude=minlongitude,
            maxlongitude=maxlongitude,
            level="network",
        )

        network_codes = [net.code for net in networks]
        print("Processing provider:", provider, "with networks:", len(network_codes))

        for net_code in network_codes:
            inv = client.get_stations(
                network=net_code,
                starttime=starttime,
                endtime=endtime,
                level="channel",
                minlatitude=minlatitude,
                maxlatitude=maxlatitude,
                minlongitude=minlongitude,
                maxlongitude=maxlongitude,
            )

            for network in inv:
                for station in network:
                    lat = float(station.latitude)
                    lon = float(station.longitude)

                    if nz_coast is not None and not is_point_inside_nz(
                        lat, lon, nz_coast=nz_coast
                    ):
                        continue

                    current_channels: set[tuple[str, str]] = set()
                    for channel in station:
                        chan_id = (channel.location_code, channel.code[:2])
                        if chan_id in current_channels:
                            continue

                        current_channels.add(chan_id)
                        all_station_info.append(
                            [
                                provider,
                                network.code,
                                station.code,
                                lat,
                                lon,
                                getattr(station, "elevation", None),
                                channel.code[:2],
                                channel.location_code,
                                getattr(channel, "start_date", None),
                                getattr(channel, "end_date", None),
                            ]
                        )

    return all_station_info


def build_station_dataframe(rows: Iterable[list[object]]) -> pd.DataFrame:
    """Build the station/channel dataframe.

    Parameters
    ----------
    rows : Iterable[list[object]]
        Station/channel rows.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing station/channel metadata.
    """
    station_df = pd.DataFrame(
        list(rows),
        columns=[
            "provider",
            "net",
            "sta",
            "lat",
            "lon",
            "elev",
            "chan",
            "loc",
            "start_date",
            "end_date",
        ],
    )

    if station_df.empty:
        return station_df

    return station_df.drop_duplicates(
        ["provider", "net", "sta", "chan", "loc"]
    ).reset_index(drop=True)


def write_station_csv(df: pd.DataFrame, out_csv: Path) -> None:
    """Write station/channel metadata to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write.
    out_csv : Path
        Output file path.

    Returns
    -------
    None
        This function returns ``None``.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


@app.command()
def get_stations(
    out_all_csv: Annotated[
        Path,
        typer.Argument(
            dir_okay=False,
            help="Output CSV for all channels.",
        ),
    ],
    out_filtered_csv: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False,
            help=(
                "Optional output CSV for desired channels only. If omitted, the filtered file is not written."
            ),
        ),
    ] = None,
    provider: Annotated[
        list[str],
        typer.Option(
            "--provider",
            help="Provider(s) to query. Repeatable. If omitted, queries all known providers.",
        ),
    ] = None,
    starttime: Annotated[
        str,
        typer.Option(help="Start time for station metadata (e.g. 2000-01-01)."),
    ] = "2000-01-01",
    endtime: Annotated[
        str,
        typer.Option(help="End time for station metadata (or 'now')."),
    ] = "now",
    min_latitude: Annotated[float, typer.Option(help="Minimum latitude.")] = -49.0,
    max_latitude: Annotated[float, typer.Option(help="Maximum latitude.")] = -32.0,
    min_longitude: Annotated[float, typer.Option(help="Minimum longitude.")] = 165.0,
    max_longitude: Annotated[float, typer.Option(help="Maximum longitude.")] = -176.9,
    coastline_filter: Annotated[
        bool,
        typer.Option(
            "--coastline-filter/--no-coastline-filter",
            help="Filter to stations inside NZ coastline polygons.",
        ),
    ] = True,
    desired_channels: Annotated[
        list[str],
        typer.Option(
            "--desired-channel",
            help="Desired 2-char channel prefixes for the filtered output. Repeatable.",
        ),
    ] = None,
) -> None:
    """Query station/channel metadata and write CSV outputs.

    Parameters
    ----------
    out_all_csv : Path
        Output CSV for all station/channel rows.
    out_filtered_csv : Path | None
        Optional output CSV for filtered station/channel rows.
    provider : list[str], optional
        Provider(s) to query. Repeatable.
    starttime : str
        Start time for station metadata.
    endtime : str
        End time for station metadata.
    min_latitude : float
        Minimum latitude.
    max_latitude : float
        Maximum latitude.
    min_longitude : float
        Minimum longitude.
    max_longitude : float
        Maximum longitude.
    coastline_filter : bool
        If True, filter to stations in NZ coastline polygons.
    desired_channels : list[str], optional
        Desired 2-character channel prefixes for ``out_filtered_csv``.
    """
    providers = _iter_providers(provider)

    rows = collect_station_rows(
        providers=providers,
        starttime=_parse_time(starttime),
        endtime=_parse_time(endtime),
        minlatitude=min_latitude,
        maxlatitude=max_latitude,
        minlongitude=min_longitude,
        maxlongitude=max_longitude,
        coastline_filter=coastline_filter,
    )

    station_df = build_station_dataframe(rows)
    write_station_csv(station_df, out_all_csv)

    if out_filtered_csv is None:
        return

    desired = desired_channels if desired_channels else ["HH", "BH", "HN", "BN"]
    filtered_df = station_df[station_df["chan"].isin(desired)].reset_index(drop=True)
    write_station_csv(filtered_df, out_filtered_csv)


if __name__ == "__main__":
    app()
