"""
    Contains the functions to add tectonic class to the data
"""

from pathlib import Path

import numpy as np
import pandas as pd
import fiona
import multiprocessing
from functools import partial
from typing import Tuple

from qcore import geo


def merge_NZSMDB_flatfile_on_events(
    event_df: pd.DataFrame,
    NZSMDB_csv_path: Path,
):
    """
    Merge metadata fields from NZ SMDB flatfile (Van Houtte, 2017)

    Parameters
    ----------
    event_df : pd.DataFrame
        The event dataframe
    NZSMDB_csv_path : Path
        The path to the NZSMDB csv file
    """
    event_cols = [
        "CuspID",
        "Origin_time",
        "Mw",
        "MwUncert",
        "TectClass",
        "Mech",
        "PreferredFaultPlane",
        "Strike",
        "Dip",
        "Rake",
        "Location",
        "HypLat",
        "HypLon",
        "HypN",
        "HypE",
        "LENGTH_km",
        "WIDTH_km",
    ]

    NZSMDB_df = pd.read_csv(NZSMDB_csv_path).drop_duplicates(subset=["CuspID"])
    new_columns = {col: f"NZSMDB_{col}" for col in event_cols}
    NZSMDB_df = NZSMDB_df.rename(columns=new_columns)
    event_df = event_df.merge(
        right=NZSMDB_df[new_columns.values()],
        how="left",
        left_on="evid",
        right_on="NZSMDB_CuspID",
    )
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
    """
    # Manage index and column renaming
    event_df.set_index('evid', inplace=True)
    cmt_df.set_index('PublicID', inplace=True)
    cmt_df.rename(columns={'Mw': 'mag', 'Latitude': 'lat', 'Longitude': 'lon', 'CD': 'depth'}, inplace=True)

    # Get the intersection of indices (evid and PublicID)
    common_ids = event_df.index.intersection(cmt_df.index)

    # Update values in event df from the CMT data where the evid is the same
    event_df.loc[common_ids, ['mag', 'lat', 'lon', 'depth']] = cmt_df.loc[
        common_ids, ['mag', 'lat', 'lon', 'depth']]
    event_df.loc[common_ids, ['mag_type', 'mag_method', 'loc_type', 'loc_grid']] = "Mw", "CMT", "CMT", "CMT"

    # Reset index
    event_df.reset_index(inplace=True)

    return event_df


def xyz_fault_points(
    fault_file: Path,
    d_s: float,
    d_d: float,
    R_a: dict = None,
    R_b: dict = None,
    R_c: dict = None,
):
    """Determine an array of points on, and offshore, of each fault
    (PEER NGA SUB, 2020)

    Parameters
    ----------
    fault_file (Path): Text file of longitude, latitude, and depth (-)
    d_s (int,float): Upper limit of seismogenic zone, Hayes, 2018
    d_d (int,float): Lower limit of seismogenic zone, Hayes, 2018
    R_a (dict): portion of fault in region a (lat,long,depth) NGA-SUB, 2020 (optional)
    R_b (dict): portion of fault in region b (lat,long,depth) NGA-SUB, 2020 (optional)
    R_c (dict): portion of fault in region c (lat,long,depth) NGA-SUB, 2020 (optional)

    Returns
    -------
    R_a (dict): portion of faults in region a (lat,long,depth) NGA-SUB, 2020
    R_b (dict): portion of faults in region b (lat,long,depth) NGA-SUB, 2020
    R_c (dict): portion of faults in region c (lat,long,depth) NGA-SUB, 2020

    """
    # Read fault file
    df = pd.read_csv(fault_file, sep=",", engine="python", header=None)
    df.dropna(how="any", inplace=True)

    # Rename columns
    col_dict = {df.columns[-3]: "long", df.columns[-2]: "lat", df.columns[-1]: "depth"}
    df.rename(columns=col_dict, inplace=True)

    # Ensure positive longitudes and depths
    df["depth"] = df["depth"].abs()
    df["long"] = df["long"].abs()

    # Divide into regions
    df_a = df[(df.depth < d_s)]
    df_b = df[(df.depth >= d_s) & (df.depth <= d_d)]
    df_c = df[(df.depth > d_d)]

    # Initialize dictionaries if they are None
    R_a = R_a or {}
    R_b = R_b or {}
    R_c = R_c or {}

    # Add fault to fault surface dictionary
    fault = fault_file.stem
    R_a[fault] = df_a.to_numpy()
    R_b[fault] = df_b.to_numpy()
    R_c[fault] = df_c.to_numpy()

    # NOTE: ALL BELOW IS NOT NEEDED IF R_a_syn IS NOT NEEDED
    # Project additional default offshore region to add to R_a_syn dictionary
    # Get centre of fault surface definition
    # pt_1 = df["long"].idxmax(axis=0)
    # pt_2 = df["long"].idxmin(axis=0)
    # pt_3 = df["lat"].idxmax(axis=0)
    # pt_4 = df["lat"].idxmin(axis=0)
    #
    # # Calculate midpoint 1
    # midpoint_1 = geo.ll_mid(df.long[pt_1], df.lat[pt_1], df.long[pt_2], df.lat[pt_2])
    #
    # # Calculate midpoint 2
    # midpoint_2 = geo.ll_mid(df.long[pt_3], df.lat[pt_3], df.long[pt_4], df.lat[pt_4])
    #
    # # Calculate fault surface midpoint
    # lonc_surface, latc_surface = geo.ll_mid(midpoint_1[0], midpoint_1[1], midpoint_2[0], midpoint_2[1])
    #
    # # Isolate an approximate updip edge at d_s
    # df_updip_edge = df[(round(df.depth, 1) == d_s)]
    #
    # # Pick opposite ends of the updip edge
    # pt_a = df_updip_edge["long"].idxmax(axis=0)
    # pt_b = df_updip_edge["long"].idxmin(axis=0)
    #
    # # Get centre of the approximate updip edge
    # lonc_updip, latc_updip = geo.ll_mid(
    #     df_updip_edge.long[pt_a],
    #     df_updip_edge.lat[pt_a],
    #     df_updip_edge.long[pt_b],
    #     df_updip_edge.lat[pt_b],
    # )
    #
    # # Determine strike bearing (arbitrary asimuth)
    # strike_bearing = geo.ll_bearing(
    #     df_updip_edge.long[pt_a],
    #     df_updip_edge.lat[pt_a],
    #     df_updip_edge.long[pt_b],
    #     df_updip_edge.lat[pt_b],
    # )
    #
    # # Determine length along strike at updip edge
    # strike_length = geo.ll_dist(
    #     df_updip_edge.long[pt_a],
    #     df_updip_edge.lat[pt_a],
    #     df_updip_edge.long[pt_b],
    #     df_updip_edge.lat[pt_b],
    # )
    #
    # # Get distance from centre of updip edge to furtherest fault point
    # # For computing the offshore distance
    # corner_distances = [
    #     geo.ll_dist(df.long[pt], df.lat[pt], lonc_updip, latc_updip)
    #     for pt in [pt_1, pt_2, pt_3, pt_4]
    # ]
    #
    # # Take the offshore distance as the average distance to the two furthest corners
    # # of the fault surface from the center of the updip strike
    # os_dist = np.mean(sorted(corner_distances, reverse=True)[:2])
    #
    # # Determine the bearing
    # updip_center_bearing = geo.ll_bearing(
    #     lonc_updip,
    #     latc_updip,
    #     lonc_surface,
    #     latc_surface,
    # )
    #
    # half_range = 90
    # upper_limit = (updip_center_bearing + half_range) % 360
    # lower_limit = (updip_center_bearing - half_range) % 360
    #
    # # Set offshore bearing as normal to strike bearing
    # # and in opposite hemisphere as updip-center bearing
    # for pot_offshore_bearing in [strike_bearing - 90, strike_bearing + 90]:
    #     if pot_offshore_bearing > upper_limit or pot_offshore_bearing < lower_limit:
    #         offshore_bearing = pot_offshore_bearing
    #
    # # Points in offshore direction
    # os_vect = np.linspace(
    #     0,
    #     os_dist,
    #     int(np.ceil(os_dist / grid_space)),
    # )
    #
    # # Points in along-strike direction
    # y_vect = np.linspace(
    #     -strike_length / 2,
    #     strike_length / 2,
    #     int(np.ceil(strike_length / grid_space)),
    #     )
    #
    # os_mesh, y_mesh = np.meshgrid(os_vect, y_vect)
    # osy_arr = np.column_stack((os_mesh.ravel(order='F'), y_mesh.ravel(order='F')))
    #
    # amat_os, _ = geo.gen_mat(offshore_bearing, lonc_updip, latc_updip)
    # ll_arr = geo.xy2ll(osy_arr, amat_os)
    # depth_arr = np.zeros((ll_arr.shape[0], 1))
    #
    # R_a_syn[fault] = np.concatenate(
    #     (ll_arr, depth_arr),
    #     axis=1,
    # )

    return R_a, R_b, R_c


def ngasub2020_tectclass_v3(
    row_tuple: Tuple[int, pd.Series],
    R_a: dict = False,
    R_b: dict = False,
    R_c: dict = False,
    fault_label: str = np.nan,
    h_thresh: float =10,
    v_thresh: float =10,
):
    """Applies the modified classification logic from the NGA-SUB 2020 report
    (PEER NGA SUB, 2020)

    Region A (vertical prism offshore of seismogenic zone of fault plane):
    depth<60km: 'Outer rise'
    depth>=60km: 'Slab'

    Region B (vertical prism containing seismogenic zone of fault plane):
    depth<min(slab surface, 20km): 'Crustal'
    min(slab surface, 20km)>depth>60km: 'Interface'
    depth>60km: 'Slab'

    Region C (vertical prism downdip of the seismogenic zone of the fault plane):
    depth<30km: 'Crustal'
    30km<depth<slab surface: 'Undetermined'
    depth>slab surface: 'Slab'

    Elsewhere (Farfield):
    depth<30km: 'Crustal'
    depth>30km: 'Undetermined'


    Parameters
    ----------
    row_tuple (tuple[int, pd.Series]): The event row
    R_a (dict): portion of faults in region a (lat,long,depth) NGA-SUB, 2020
    R_b (dict): portion of faults in region b (lat,long,depth) NGA-SUB, 2020
    R_c (dict): portion of faults in region c (lat,long,depth) NGA-SUB, 2020
    fault_label (str): The fault label
    h_thresh (float): Horizontal distance threshold
    v_thresh (float): Vertical distance threshold

    Returns
    -------
    tectclass (str): 'Slab', 'Interface', 'Outer rise', 'Crustal', or 'Undetermined'
    fault (str): fault which triggered 'Interface','Slab' or 'Outer rise' tectclass labels
    """
    row = row_tuple[1]
    lon, lat, depth = row["lat"], row["lon"], row["depth"]

    # Initially classify as if farfield, correct later if neccessary
    if depth <= 30:
        tectclass = "Crustal"
    elif depth > 60:
        tectclass = "Slab"
    else:
        tectclass = "Undetermined"

    for flag, region in zip(["A", "C", "B"], [R_a, R_c, R_b]):
        for fault, pt_arr in region.items():
            i, d = geo.closest_location(pt_arr[:, :2], lon, lat)

            if d < h_thresh:
                fault_label = fault
                # Classifications for region A
                if flag == "A":
                    if depth <= 60:
                        tectclass = "Outer-rise"
                    else:
                        tectclass = "Slab"

                # Classifications for region B
                elif flag == "B":
                    if depth <= region[fault][i][-1] - v_thresh and depth <= 20:
                        tectclass = "Crustal"
                    elif depth <= 60 and depth <= region[fault][i][-1] + v_thresh:
                        tectclass = "Interface"
                    else:
                        tectclass = "Slab"

                # Classifications for region C
                elif flag == "C":
                    if depth <= 30:
                        tectclass = "Crustal"
                    elif depth >= region[fault][i][-1] - v_thresh:
                        tectclass = "Slab"
                    else:
                        tectclass = "Undetermined"

    return tectclass, fault_label


def filter_tectclass(event_df, tectclass_df, cmt_tectclass_df):
    cmt_tectclass_df.rename(
        columns={"PublicID": "evid", "tectclass": "tect_class"}, inplace=True
    )
    cmt_tectclass_df["tect_method"] = "manual"

    tectclass_df["tect_class"] = tectclass_df["NGASUB_TectClass_Merged"]
    tectclass_df["tect_method"] = tectclass_df["NGASUB_Faults_Merged"]
    tectclass_df.loc[tectclass_df.NZSMDB_TectClass.isnull() == False, "tect_class"] = (
        tectclass_df[
            tectclass_df.NZSMDB_TectClass.isnull() == False
        ].NZSMDB_TectClass.values
    )
    tectclass_df.loc[tectclass_df.NZSMDB_TectClass.isnull() == False, "tect_method"] = (
        "NZSMDB"
    )
    # merged_df = event_df.set_index('evid').join(tectclass_df[['evid','tect_class','tect_method']].set_index('evid'),how='left').reset_index()
    merged_df = event_df
    merged_df = (
        merged_df.set_index("evid")
        .join(
            tectclass_df[["evid", "tect_class", "tect_method"]].set_index("evid"),
            how="left",
            rsuffix="_redone",
        )
        .reset_index()
    )
    if "tect_class_redone" in merged_df.columns:
        merged_df[["tect_class", "tect_method"]] = merged_df[
            ["tect_class_redone", "tect_method_redone"]
        ]
    merged_df = (
        merged_df.set_index("evid")
        .join(
            cmt_tectclass_df[["evid", "tect_class", "tect_method"]].set_index("evid"),
            how="left",
            rsuffix="_manual",
        )
        .reset_index()
    )
    merged_df.loc[
        ~merged_df.tect_class_manual.isnull(), ["tect_class", "tect_method"]
    ] = merged_df.loc[
        ~merged_df.tect_class_manual.isnull(),
        ["tect_class_manual", "tect_method_manual"],
    ].values
    if "tect_class_redone" in merged_df.columns:
        merged_df.drop(
            columns=[
                "tect_class_redone",
                "tect_method_redone",
                "tect_class_manual",
                "tect_method_manual",
            ],
            inplace=True,
        )
    else:
        merged_df.drop(
            columns=["tect_class_manual", "tect_method_manual"], inplace=True
        )

    return merged_df


def add_tect_class(cmt_tectclass_ffp: Path, tect_shape_ffp: Path, geonet_cmt_ffp: Path, event_csv_ffp: Path, NZ_SMDB_path: Path, sub_surface_dir: Path):
    """
    Adds the tectonic class to the event data
    """
    # Read the CMT tectonic class data
    cmt_tectclass_df = pd.read_csv(cmt_tectclass_ffp, low_memory=False)

    # Shape file for determining neotectonic domain
    shape = fiona.open(tect_shape_ffp)

    # Read the geonet CMT and event data
    geonet_cmt_df = pd.read_csv(geonet_cmt_ffp, low_memory=False)
    event_df = pd.read_csv(event_csv_ffp, low_memory=False).set_index("evid").reset_index()

    # Replace the geonet CMT data on the event data
    event_df = replace_cmt_data_on_event(event_df, geonet_cmt_df)

    # Merge the NZSMDB data
    event_df = merge_NZSMDB_flatfile_on_events(event_df, NZ_SMDB_path)

    # Merge Kermadec and Hikurangi datasets
    merged_a, merged_b, merged_c = xyz_fault_points(
            fault_file=sub_surface_dir / "Merged_slab/hik_kerm_fault_300km_wgs84_poslon.txt",
            d_s=10,  # Hayes et al., 2018
            d_d=47,  # Hayes et al., 2018
        )

    # Merge Puysegur dataset
    merged_a, merged_b, merged_c = xyz_fault_points(
        fault_file=sub_surface_dir / "Slab2_2018/puy/puy_slab2_dep_02.26.18.xyz",
        d_s=11,  # Hayes et al., 2018
        d_d=30,  # Hayes et al., 2018
        R_a=merged_a,
        R_b=merged_b,
        R_c=merged_c,
    )

    # Create a pool of workers
    pool = multiprocessing.Pool(processes=8)

    # Apply the function to each row of the DataFrame in parallel
    results = pool.map(
        partial(ngasub2020_tectclass_v3, R_a=merged_a, R_b=merged_b, R_c=merged_c),
        event_df.iterrows(),
    )

    # Close the pool of workers
    pool.close()
    pool.join()

    # Assign the results to the respective DataFrame columns
    event_df["NGASUB_TectClass_Merged"], event_df["NGASUB_Faults_Merged"] = zip(*results)

    # Merge tectonic classification data from both CMT and regular event data
    merged_df = filter_tectclass(event_df, df, cmt_tectclass_df)


add_tect_class(
    Path("/home/joel/code/nzgmdb/archive/focal/GeoNet-v04-tectclass.csv"),
    Path("/home/joel/code/nzgmdb/archive/TectonicDomains/TectonicDomains_Feb2021_8_NZTM.shp"),
    Path("/home/joel/code/nzgmdb/archive/focal/GeoNet_CMT_solutions.csv"),
    Path("/home/joel/local/gmdb/new_data_walkthrough/archive/earthquake_source_table.csv"),
    Path("/home/joel/local/gmdb/tect_domain_folders/Records/NZ_SMDB/Spectra_flatfiles/NZdatabase_flatfile_FAS_horizontal_GeoMean.csv"),
    Path("/home/joel/local/gmdb/tect_domain_folders/geospatial/Subduction_surfaces"),
)
