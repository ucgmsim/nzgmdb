import fiona
import pandas as pd
import numpy as np
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.management import file_structure, config as cfg
from nzgmdb.data_retrieval import tect_domain
from qcore import point_in_polygon
from Velocity_Model.basins import basin_outlines_dict


def create_site_table_response() -> pd.DataFrame:
    """
    Create the site table for the NZGMDB. This function fetches the station information from the FDSN clients, and the
    Geonet metadata summary information. It then merges the two dataframes and determines the tectonic domain for each
    station. The final dataframe is saved as a csv file in the flatfile directory.

    Returns
    -------
    pd.DataFrame
        The site table dataframe with all Z, vs30, domain and location values for each site
        used in the NZGMDB
    """
    # Fetch the client station information
    client_NZ = FDSN_Client("GEONET")
    client_IU = FDSN_Client("IRIS")
    inventory_NZ = client_NZ.get_stations()
    inventory_IU = client_IU.get_stations(network="IU", station="SNZO")
    inventory = inventory_NZ + inventory_IU
    station_info = []
    for network in inventory:
        for station in network:
            station_info.append(
                [
                    network.code,
                    station.code,
                    station.latitude,
                    station.longitude,
                    station.elevation,
                ]
            )
    sta_df = pd.DataFrame(station_info, columns=["net", "sta", "lat", "lon", "elev"])
    sta_df = sta_df.drop_duplicates().reset_index(drop=True)

    # Get the Geonet metadata summary information
    data_dir = file_structure.get_data_dir()
    geo_meta_summary_df = pd.read_csv(data_dir / "Geonet  Metadata  Summary_v1.4.csv")

    # Rename the columns
    geo_meta_summary_df = geo_meta_summary_df.rename(
        columns={
            "Name": "sta",
            "Lat": "lat",
            "Long": "lon",
            "NZS1170SiteClass": "site_class",
            "Vs30_median": "Vs30",
            "Sigmaln_Vs30": "Vs30_std",
            "T_median": "T0",
            "sigmaln_T": "T0_std",
            "Q_T": "Q_T0",
            "D_T": "D_T0",
            "T_Ref": "T0_ref",
            "Z1.0_median": "Z1.0",
            "sigmaln_Z1.0": "Z1.0_std",
            "Z1.0_Ref": "Z1.0_ref",
            "Z2.5_median": "Z2.5",
            "sigmaln_Z2.5": "Z2.5_std",
            "Z2.5_Ref": "Z2.5_ref",
        }
    )

    merged_df = geo_meta_summary_df.merge(
        sta_df[["net", "elev", "sta"]], on="sta", how="left"
    )
    # Shape file for determining domain
    shapes = list(
        fiona.open(data_dir / "tect_domain" / "TectonicDomains_Feb2021_8_NZTM.shp")
    )
    tect_merged_df = tect_domain.find_domain_from_shapes(merged_df, shapes)

    # Rename the domain column
    tect_merged_df = tect_merged_df.rename(columns={"domain_no": "site_domain_no"})

    # Select specific columns
    site_df = tect_merged_df[
        [
            "net",
            "sta",
            "lat",
            "lon",
            "elev",
            "site_class",
            "Vs30",
            "Vs30_std",
            "Q_Vs30",
            "Vs30_Ref",
            "T0",
            "T0_std",
            "Q_T0",
            "D_T0",
            "T0_ref",
            "Z1.0",
            "Z1.0_std",
            "Q_Z1.0",
            "Z1.0_ref",
            "Z2.5",
            "Z2.5_std",
            "Q_Z2.5",
            "Z2.5_ref",
            "site_domain_no",
        ]
    ]
    site_df = site_df.astype({"Z2.5": float})
    site_df.loc[:, "Z2.5"] /= 1000.0

    return site_df


def add_site_basins(site_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the site basins to the site table

    Parameters
    ----------
    site_df : pd.DataFrame
        The site table dataframe with at least the columns 'lon' and 'lat'
        Ideally and in most cases, this dataframe should be the output of create_site_table_response

    Returns
    -------
    pd.DataFrame
        The site dataframe with the 'basin' column added
    """
    # Get the site table and points
    ll_points = site_df[["lon", "lat"]].values
    site_df["basin"] = None

    # Define rename basins
    rename_dict = {
        "NewCanterburyBasinBoundary": "Canterbury",
        "BPVBoundary": "Banks Peninsula volcanics",
        "waitaki": "Waitaki",
        "Napier1": "Napier",
        "mackenzie": "Mackenzie",
        "NorthCanterbury": "North Canterbury",
        "dun": "Dun",
        "WakatipuBasinOutlineWGS84": "Wakatipu",
        "WaikatoHaurakiBasinEdge": "Waikato Hauraki",
        "HawkesBay1": "Hawkes Bay",
        "WanakaOutlineWGS84": "Wanaka",
        "Porirua1": "Porirua",
        "SpringsJ": "Springs Junction",
        "CollingwoodBasinOutline": "Collingwood",
        "GreaterWellington4": "Greater Wellington",
    }

    # Get the basin version
    config = cfg.Config()
    version = config.get_value("basin_version")

    # Get the basin outlines
    basin_outlines = basin_outlines_dict[version]

    for cur_ffp in basin_outlines:
        # Get the basin name and its rename
        basin_name = cur_ffp.stem.split("_")[0].split(".")[0]
        basin_name = rename_dict.get(basin_name, basin_name)

        # Get the outline
        basin_outline = np.loadtxt(cur_ffp)

        # Find sites within basin
        is_inside_basin = point_in_polygon.is_inside_postgis_parallel(
            ll_points, basin_outline
        )
        site_df.loc[is_inside_basin, "basin"] = basin_name

    return site_df
