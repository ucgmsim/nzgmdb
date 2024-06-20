from pathlib import Path

import fiona
import pandas as pd
import numpy as np
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.management import file_structure, config as cfg
from nzgmdb.data_retrieval import tect_domain
from qcore import point_in_polygon
# from Velocity_Model.basins import basin_outlines_dict


def create_site_table_response(main_dir: Path):
    """
    Create the site table for the GMDB. This function fetches the station information from the FDSN clients, and the
    Geonet metadata summary information. It then merges the two dataframes and determines the tectonic domain for each
    station. The final dataframe is saved as a csv file in the flatfile directory.

    Parameters
    ----------
    main_dir : Path
        The main directory to the NZGMDB results (Highest level directory)
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
            "T_median": "Tsite",
            "sigmaln_T": "Tsite_std",
            "Q_T": "Q_Tsite",
            "D_T": "D_Tsite",
            "T_Ref": "Tsite_ref",
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
            "Tsite",
            "Tsite_std",
            "Q_Tsite",
            "D_Tsite",
            "Tsite_ref",
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
    site_df["Z2.5"] /= 1000

    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    site_df.to_csv(flatfile_dir / "site_table_basin.csv", index=False)


def add_site_basins(main_dir: Path):
    """
    Add the site basins to the site table

    Parameters
    ----------
    main_dir : Path
        The main directory to the NZGMDB results (Highest level directory)
    """
    data_dir = file_structure.get_data_dir()
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Get the site table and points
    site_table = pd.read_csv(flatfile_dir / "site_table_basin.csv")
    ll_points = site_table[["lon", "lat"]].values
    site_table["basin"] = None

    # Get all the basin versions
    config = cfg.Config()
    versions = config.get_value("basin_versions")
    for version in versions:
        # Get the basin outlines and load them into boundaries
        basin_outlines = basin_outlines_dict[version]
        for cur_ffp in basin_outlines:
            basin_name = cur_ffp.stem
            basin_outline = np.loadtxt(cur_ffp)
            # Find sites within basin
            is_inside_basin = point_in_polygon.is_inside_postgis_parallel(
                np.flip(ll_points, axis=1), basin_outline
            )  # flip the coordinates to (lon, lat) format
            site_table[is_inside_basin] = basin_name

    # Rename basins TODO

    site_table.to_csv(flatfile_dir / "site_table_basin.csv", index=False)




# create_site_table_response(Path("/home/joel/local/gmdb/US_stuff/new_struct_2022"))
# add_site_basins(Path("/home/joel/local/gmdb/US_stuff/new_struct_2022"))