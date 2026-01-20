from obspy.clients.fdsn import Client as FDSN_Client
from obspy import UTCDateTime

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

# Define rough NZ bounding box (adjust as needed)
min_lat, max_lat = -49, -32.0
min_lon, max_lon = 165.0, -176.9

# Time window for station metadata
starttime = UTCDateTime("2000-01-01")
endtime = UTCDateTime()  # now

import pandas as pd

all_station_info = []

import nzgeom.coastlines
from shapely.geometry import Point

# Load NZ coastline polygons once (efficient)
_NZ_COAST = nzgeom.coastlines.get_NZ_coastlines().to_crs("EPSG:4326")


def is_point_inside_nz(lat, lon):
    """
    Returns True if the given latitude/longitude lies inside
    the NZ mainland or island coastline polygons.
    """
    # shapely uses (lon, lat)
    p = Point(lon, lat)
    # test containment against all polygons
    return _NZ_COAST.geometry.apply(lambda g: g.contains(p)).any()


for provider, _ in URL_MAPPINGS.items():
    try:
        client = FDSN_Client(base_url=provider)
        networks = client.get_stations(
            starttime=starttime,
            endtime=endtime,
            minlatitude=min_lat,
            maxlatitude=max_lat,
            minlongitude=min_lon,
            maxlongitude=max_lon,
            level="network",
        )
        network_codes = [net.code for net in networks]

        print("Processing provider:", provider, "with networks:", len(network_codes))

        for net_code in network_codes:
            try:
                inv = client.get_stations(
                    network=net_code,
                    level="channel",
                    minlatitude=min_lat,
                    maxlatitude=max_lat,
                    minlongitude=min_lon,
                    maxlongitude=max_lon,
                )

                for network in inv:
                    print(
                        "  Network:",
                        network.code,
                        "with stations:",
                        len(network.stations),
                    )

                    for station in network:
                        lat = station.latitude
                        lon = station.longitude
                        if not is_point_inside_nz(lat, lon):
                            continue
                        current_channels = set()
                        for channel in station:
                            chan_id = (channel.location_code, channel.code[:2])
                            if chan_id in current_channels:
                                continue
                            current_channels.add(chan_id)
                            all_station_info.append(
                                [
                                    provider,  # provider as first column
                                    network.code,
                                    station.code,
                                    lat,
                                    lon,
                                    station.elevation,
                                    channel.code[:2],
                                    channel.location_code,
                                    channel.start_date,
                                    channel.end_date,
                                ]
                            )
            except Exception:
                # continue to next network code on failure
                continue
    except Exception:
        # continue to next provider on failure
        continue

# build dataframe (provider is first column)
station_df = pd.DataFrame(
    all_station_info,
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
station_df = station_df.drop_duplicates(
    ["provider", "net", "sta", "chan", "loc"]
).reset_index(drop=True)

# write outputs
station_df.to_csv(
    "/media/joel/data/nzgmdb/tmp_arrays/nz_mainland_stations_all_provider_networks_channels.csv",
    index=False,
)

desired_channels = ["HH", "BH", "HN", "BN"]
filtered_df = station_df[station_df["chan"].isin(desired_channels)]
filtered_df.to_csv(
    "/media/joel/data/nzgmdb/tmp_arrays/nz_mainland_stations_all_provider_networks_desired_channels.csv",
    index=False,
)
