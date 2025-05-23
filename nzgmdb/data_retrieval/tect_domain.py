"""
Contains the functions to add tectonic domain to the data
"""

import multiprocessing
from functools import partial
from pathlib import Path

import fiona
import numpy as np
import pandas as pd
from pyproj import Transformer

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from qcore import geo, point_in_polygon


def merge_NZSMDB_flatfile_on_events(
    event_df: pd.DataFrame,
):
    """
    Merge metadata fields from NZ SMDB flatfile (Van Houtte, 2017)

    Parameters
    ----------
    event_df : pd.DataFrame
        The event dataframe

    Returns
    -------
    pd.DataFrame
        The event dataframe with the NZSMDB data merged
    """
    nzsmdb_cols = [
        "CuspID",
        "TectClass",
    ]
    config = cfg.Config()
    NZSMDB_df = pd.read_csv(config.get_value("nzsmdb_url")).drop_duplicates(
        subset=["CuspID"]
    )
    new_columns = {col: f"NZSMDB_{col}" for col in nzsmdb_cols}
    NZSMDB_df = NZSMDB_df.rename(columns=new_columns)
    event_df = event_df.merge(
        right=NZSMDB_df[new_columns.values()],
        how="left",
        left_on="evid",
        right_on="NZSMDB_CuspID",
    )
    # Drop the NZSMDB_CuspID column
    event_df = event_df.drop(columns=["NZSMDB_CuspID"])
    return event_df


def replace_cmt_data_on_event(
    event_df: pd.DataFrame,
    cmt_df: pd.DataFrame,
):
    """
    Replace the event data with the CMT data where the evid is the same
    Specifically, the magnitude, latitude, longitude, depth
    Also sets the mag_type, mag_method, loc_type, loc_grid to "Mw", "CMT", "CMT", "CMT"

    Parameters
    ----------
    event_df : pd.DataFrame
        The event dataframe
    cmt_df : pd.DataFrame
        The CMT dataframe

    Returns
    -------
    pd.DataFrame
        The event dataframe with the CMT data merged
    """
    # Manage index and column renaming
    cmt_df = cmt_df.rename(
        columns={"Mw": "mag", "Latitude": "lat", "Longitude": "lon", "CD": "depth"}
    ).set_index("PublicID")

    # Get the intersection of indices (evid and PublicID)
    common_ids = event_df.index.intersection(cmt_df.index)

    # Update values in event df from the CMT data where the evid is the same
    event_df.loc[common_ids, ["mag", "lat", "lon", "depth"]] = cmt_df.loc[
        common_ids, ["mag", "lat", "lon", "depth"]
    ]
    event_df.loc[common_ids, ["mag_type", "mag_method", "loc_type", "loc_grid"]] = (
        "Mw",
        "CMT",
        "CMT",
        "CMT",
    )

    return event_df.reset_index()


def create_regions(
    fault_file: Path,
    d_s: float,
    d_d: float,
    region_a_offshore: dict = None,
    region_b_on: dict = None,
    region_c_downdip: dict = None,
):
    """
    Determine an array of points on and offshore of a fault and divide them into regions.

    The regions follow the PEER NGA SUB (2020) classification:

    - **Region A**: Vertical prism offshore of the seismogenic zone of the fault plane.
    - **Region B**: Vertical prism containing the seismogenic zone of the fault plane.
    - **Region C**: Vertical prism downdip of the seismogenic zone of the fault plane.

    Parameters
    ----------
    fault_file : Path
        Text file containing longitude, latitude, and depth values.
    d_s : float
        Upper limit of the seismogenic zone (Hayes, 2018).
    d_d : float
        Lower limit of the seismogenic zone (Hayes, 2018).
    region_a_offshore : dict, optional
        Portion of the fault in Region A (latitude, longitude, depth) as defined in NGA-SUB (2020).
    region_b_on : dict, optional
        Portion of the fault in Region B (latitude, longitude, depth) as defined in NGA-SUB (2020).
    region_c_downdip : dict, optional
        Portion of the fault in Region C (latitude, longitude, depth) as defined in NGA-SUB (2020).

    Returns
    -------
    tuple
        A tuple containing:
        - **region_a_offshore** (*dict*): Portion of faults in Region A (latitude, longitude, depth).
        - **region_b_on** (*dict*): Portion of faults in Region B (latitude, longitude, depth).
        - **region_c_downdip** (*dict*): Portion of faults in Region C (latitude, longitude, depth).

    """
    # Read fault file
    df = pd.read_csv(fault_file, sep=",", engine="python", header=None)
    df = df.dropna(how="any")

    # Rename columns
    col_dict = {df.columns[-3]: "long", df.columns[-2]: "lat", df.columns[-1]: "depth"}
    df = df.rename(columns=col_dict)

    # Ensure positive longitudes and depths
    df["depth"] = df["depth"].abs()
    df["long"] = df["long"].abs()

    # Divide into regions
    df_a = df[(df.depth < d_s)]
    df_b = df[(df.depth >= d_s) & (df.depth <= d_d)]
    df_c = df[(df.depth > d_d)]

    # Initialize dictionaries if they are None
    region_a_offshore = region_a_offshore or {}
    region_b_on = region_b_on or {}
    region_c_downdip = region_c_downdip or {}

    # Add fault to fault surface dictionary
    fault = fault_file.stem
    region_a_offshore[fault] = df_a.to_numpy()
    region_b_on[fault] = df_b.to_numpy()
    region_c_downdip[fault] = df_c.to_numpy()

    return region_a_offshore, region_b_on, region_c_downdip


def ngasub2020_tectclass(
    row: pd.Series,
    region_a_offshore: dict = False,
    region_b_on: dict = False,
    region_c_downdip: dict = False,
    fault_label: str = np.nan,
    h_thresh: float = 10,
    v_thresh: float = 10,
):
    """
    Applies the modified classification logic from the NGA-SUB 2020 report.

    The classification is based on depth and region:

    - **Region A (offshore seismogenic zone)**:
        - depth < 60 km: 'Outer rise'
        - depth >= 60 km: 'Slab'
    - **Region B (seismogenic zone)**:
        - depth < min(slab surface, 20 km): 'Crustal'
        - min(slab surface, 20 km) > depth > 60 km: 'Interface'
        - depth > 60 km: 'Slab'
    - **Region C (downdip of seismogenic zone)**:
        - depth < 30 km: 'Crustal'
        - 30 km < depth < slab surface: 'Undetermined'
        - depth > slab surface: 'Slab'
    - **Elsewhere (Farfield)**:
        - depth < 30 km: 'Crustal'
        - depth > 30 km: 'Undetermined'

    Parameters
    ----------
    row : pd.Series
        The event row containing necessary seismic information.
    region_a_offshore : dict
        Portion of faults in Region A (latitude, longitude, depth) as defined in NGA-SUB (2020).
    region_b_on : dict
        Portion of faults in Region B (latitude, longitude, depth) as defined in NGA-SUB (2020).
    region_c_downdip : dict
        Portion of faults in Region C (latitude, longitude, depth) as defined in NGA-SUB (2020).
    fault_label : str
        The fault label associated with the event.
    h_thresh : float
        Horizontal distance threshold for classification.
    v_thresh : float
        Vertical distance threshold for classification.

    Returns
    -------
    tuple
        A tuple containing:
        - **tectclass** (*str*): 'Slab', 'Interface', 'Outer rise', 'Crustal', or 'Undetermined'.
        - **fault** (*str* or *None*): The fault responsible for 'Interface', 'Slab', or 'Outer rise' tectclass labels.

    """
    lat, lon, depth = row["lat"], row["lon"], row["depth"]

    # Initially classify as if farfield, correct later if neccessary
    if depth <= 30:
        tectclass = "Crustal"
    elif depth > 60:
        tectclass = "Slab"
    else:
        tectclass = "Undetermined"

    for flag, region in zip(
        ["A", "C", "B"], [region_a_offshore, region_c_downdip, region_b_on]
    ):
        for fault, pt_arr in region.items():
            closest_index, distance = geo.closest_location(pt_arr[:, :2], lon, lat)

            if distance < h_thresh:
                fault_label = fault
                # Classifications for region A Offshore
                if flag == "A":
                    if depth <= 60:
                        tectclass = "Outer-rise"
                    else:
                        tectclass = "Slab"

                # Classifications for region B On
                elif flag == "B":
                    if (
                        depth <= region[fault][closest_index][-1] - v_thresh
                        and depth <= 20
                    ):
                        tectclass = "Crustal"
                    elif (
                        depth <= 60
                        and depth <= region[fault][closest_index][-1] + v_thresh
                    ):
                        tectclass = "Interface"
                    else:
                        tectclass = "Slab"

                # Classifications for region C DownDip
                elif flag == "C":
                    if depth <= 30:
                        tectclass = "Crustal"
                    elif depth >= region[fault][closest_index][-1] - v_thresh:
                        tectclass = "Slab"
                    else:
                        tectclass = "Undetermined"

    return tectclass, fault_label


def merge_tectclass(event_df: pd.DataFrame):
    """
    Merge the tectonic classification data from the CMT and NGASUB with the event data

    Parameters
    ----------
    event_df : pd.DataFrame
        The event dataframe which also includes the NGASUB tectonic class data

    Returns
    -------
    pd.DataFrame
        The event dataframe with the tect_class and tect_method columns updated
    """
    # Ensure the dtype of the tect_class and tect_method columns
    event_df["tect_class"] = event_df["tect_class"].astype("str")
    event_df["tect_method"] = event_df["tect_method"].astype("str")

    # Replace the tect_class and tect_method with the NZSMDB data where it exists
    mask = ~event_df.NZSMDB_TectClass.isnull()
    event_df.loc[mask, "tect_class"] = event_df.loc[mask, "NZSMDB_TectClass"].values
    event_df.loc[mask, "tect_method"] = "NZSMDB"

    # Drop unnecessary columns
    event_df = event_df.drop(columns=["NZSMDB_TectClass"])

    return event_df


def find_domain_from_shapes(
    merged_df: pd.DataFrame,
    shapes: list,
):
    """
    Find the domain from the shapes for each point in the merged_df

    Parameters
    ----------
    merged_df : pd.DataFrame
        The merged event dataframe with lat, lon data
    shapes : list
        The shapes containing the layers with domain information

    Returns
    -------
    pd.DataFrame
        The merged event dataframe with the domain_no and domain_type columns updated
    """
    # Convert the lat, lon to NZTM coordinate system
    # (https://www.linz.govt.nz/guidance/geodetic-system/coordinate-systems-used-new-zealand/projections/new-zealand-transverse-mercator-2000-nztm2000)
    config = cfg.Config()
    wgs2nztm = Transformer.from_crs(
        config.get_value("ll_num"), config.get_value("nztm_num"), always_xy=True
    )
    points = np.array(merged_df[["lon", "lat"]])
    points = np.asarray(wgs2nztm.transform(points[:, 0], points[:, 1])).T

    # Go through each domain and determine if the points are in the domain
    for layer in shapes:
        domain_no = layer["properties"]["Domain_No"]
        domain_type = layer["properties"]["DomainType"]
        geometry_type = layer["geometry"]["type"]
        geometry_coords = layer["geometry"]["coordinates"]
        in_domain = None
        if geometry_type == "MultiPolygon":
            for coords in geometry_coords:
                # Convert the coords into a numpy array
                coords = np.asarray(coords)
                in_domain_check = point_in_polygon.is_inside_postgis_parallel(
                    points, coords
                )
                # If in_domain is None, set it to the first in_domain_check
                if in_domain is None:
                    in_domain = in_domain_check
                else:
                    # If in_domain is not None, update it with the in_domain_check
                    in_domain = in_domain | in_domain_check
        else:
            # Convert the geometry coords into a numpy array
            geometry_coords = np.asarray([list(coord) for coord in geometry_coords[0]])
            in_domain = point_in_polygon.is_inside_postgis_parallel(
                points, geometry_coords
            )

        # Update the domain_no, domain_name, and domain_type based on the in_domain
        merged_df.loc[in_domain, ["domain_no", "domain_type"]] = domain_no, domain_type

    # Add Oceanic for points not in any domain
    merged_df.loc[merged_df.domain_no.isnull(), ["domain_no", "domain_type"]] = (
        0,
        "Oceanic",
    )
    return merged_df


def add_tect_domain(
    event_csv_ffp: Path,
    out_ffp: Path,
    n_procs: int = 1,
):
    """
    Adds the tectonic domain to the event data

    Parameters
    ----------
    event_csv_ffp : Path
        The path to the event data csv file
    out_ffp : Path
        The path to the output csv file
    n_procs : int
        The number of processes to use
    """
    # Get the Data folder
    data_dir = file_structure.get_data_dir()

    # Shape file for determining neotectonic domain
    shapes = list(
        fiona.open(data_dir / "tect_domain" / "TectonicDomains_Feb2021_8_NZTM.shp")
    )

    # Read the geonet CMT and event data
    config = cfg.Config()
    geonet_cmt_df = pd.read_csv(config.get_value("cmt_url"), low_memory=False)
    event_df = pd.read_csv(
        event_csv_ffp, low_memory=False, dtype={"evid": str}, index_col="evid"
    )

    # Replace the geonet CMT data on the event data
    event_df = replace_cmt_data_on_event(event_df, geonet_cmt_df)

    # Merge the NZSMDB data
    event_df = merge_NZSMDB_flatfile_on_events(event_df)

    # Merge Kermadec and Hikurangi datasets into the fault regions
    region_a_offshore, region_b_on, region_c_downdip = create_regions(
        fault_file=data_dir / "hik_kerm_fault_300km_wgs84_poslon.txt",
        d_s=10,  # Hayes et al., 2018
        d_d=47,  # Hayes et al., 2018
    )

    # Merge Puysegur dataset into the fault regions
    region_a_offshore, region_b_on, region_c_downdip = create_regions(
        fault_file=data_dir / "puy_slab2_dep_02.26.18.xyz",
        d_s=11,  # Hayes et al., 2018
        d_d=30,  # Hayes et al., 2018
        region_a_offshore=region_a_offshore,
        region_b_on=region_b_on,
        region_c_downdip=region_c_downdip,
    )

    # Apply the function to each row of the DataFrame
    f = partial(
        ngasub2020_tectclass,
        region_a_offshore=region_a_offshore,
        region_b_on=region_b_on,
        region_c_downdip=region_c_downdip,
    )
    with multiprocessing.Pool(n_procs) as pool:
        results = pool.map(f, event_df.to_records(index=False))

    # Assign the results to the respective DataFrame columns
    event_df["tect_class"], event_df["tect_method"] = zip(*results)

    # Merge tectonic classification data from NGASUB with the NZSMDB data
    merged_df = merge_tectclass(event_df)

    domain_df = find_domain_from_shapes(merged_df, shapes)

    # Save the data
    domain_df.to_csv(out_ffp, index=False)
