"""
Creates the site table for the NZGMDB. This module fetches the station information from the FDSN clients, and the
Geonet metadata summary information.
"""

from pathlib import Path

import fiona
import pandas as pd
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.data_retrieval import tect_domain
from nzgmdb.management import config as cfg
from nzgmdb.management.data_registry import NZGMDB_DATA
from qcore import point_in_polygon
from velocity_modelling import registry


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
    config = cfg.Config()
    channel_codes = ",".join(config.get_value("channel_codes"))
    inventory = client_NZ.get_stations(channel=channel_codes, level="station")
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
    geo_meta_summary_df = pd.read_csv(
        NZGMDB_DATA.fetch("Geonet_Metadata_Summary_v1.4.csv")
    )

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
    # Specify the required files for fiona
    NZGMDB_DATA.fetch("TectonicDomains_Feb2021_8_NZTM.shp")
    NZGMDB_DATA.fetch("TectonicDomains_Feb2021_8_NZTM.dbf")
    NZGMDB_DATA.fetch("TectonicDomains_Feb2021_8_NZTM.shx")

    # Shape file for determining neotectonic domain
    shapes = list(
        fiona.open(Path(NZGMDB_DATA.abspath) / "TectonicDomains_Feb2021_8_NZTM.shp")
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


def add_site_basins(site_df: pd.DataFrame, nzcvm_data_ffp: Path) -> pd.DataFrame:
    """
    Add the site basins to the site table

    Parameters
    ----------
    site_df : pd.DataFrame
        The site table dataframe with at least the columns 'lon' and 'lat'
        Ideally and in most cases, this dataframe should be the output of create_site_table_response
    nzcvm_data_ffp : Path
        The full file path to the nzcvm_data repository that stores the basin information

    Returns
    -------
    pd.DataFrame
        The site dataframe with the 'basin' column added
    """
    # Get the site table and points
    ll_points = site_df[["lon", "lat"]].values
    site_df["basin"] = None

    # Get the NZCVM version
    config = cfg.Config()
    nzcvm_version = config.get_value("nzcvm_version")
    priority_basins = config.get_value("priority_basins")

    # Create the CVMRegistry object
    registry_path = nzcvm_data_ffp / "nzcvm_registry.yaml"
    cvm_registry = registry.CVMRegistry(nzcvm_version, nzcvm_data_ffp, registry_path)

    # Make a new basin_dist from the registry
    basin_dict = {
        basin["name"].split("_")[0]: basin["boundaries"]
        for basin in cvm_registry.registry["basin"]
    }

    for basin in cvm_registry.global_params["basins"]:
        # Get the basin name
        basin_name = basin.split("_")[0]

        # Get the boundaries
        boundaries = basin_dict[basin_name]
        for boundary in boundaries:
            basin_outline = cvm_registry.load_basin_boundary(boundary)

            # Find sites within basin
            is_inside_basin = point_in_polygon.is_inside_postgis_parallel(
                ll_points, basin_outline
            )
            # Ensure we only update the basin of a site if it either doesn't have a basin or is in a priority basin
            mask_has_basin = site_df["basin"].notna()
            mask_priority = mask_has_basin & (basin_name in priority_basins)
            mask_no_basin = ~mask_has_basin
            mask_update = (
                is_inside_basin & mask_no_basin | mask_priority & is_inside_basin
            )
            site_df.loc[mask_update, "basin"] = basin_name

    # Add the nzcvm_version column
    site_df["nzcvm_version"] = nzcvm_version

    return site_df
