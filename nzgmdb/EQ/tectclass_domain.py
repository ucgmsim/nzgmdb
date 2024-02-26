# MRD, 20200331
# Python module to do visualization and analysis of actual and predicted labels
#
# This version filters the tectonic classifications, adds CMT tectonic classifications,
# and computes the tectonic domain of the events.
#
# Edited, Jesse Hutchinson, Aug. 19 2021

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Import standard packages and modules
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import shutil
from datetime import timedelta, datetime
import glob
from math import radians, cos, sin, asin, sqrt
from pandas import HDFStore
from pathlib import Path
import sys, os
import csv
import scipy
from scipy.fftpack import fft
from scipy import interpolate
from pylab import *
from pandarallel import pandarallel  # conda install -c bjrn pandarallel
import fiona  # conda install fiona
from pyproj import Transformer  # conda install pyproj

# Import locally saved packages and modules
from qcore import geo

# from data_vis import *

# Modify matplotlib settings using mplstyle file
# plt.style.use('labelvis')

# Get root directory, output path, and set global variables
cwd = os.getcwd()

# out_path = cwd + '/out/relocated'
out_path = "../output"
if not os.path.exists(out_path):
    os.makedirs(out_path)

NZ_SMDB_path = "/".join(
    cwd.split("/")[:-2]
    + [
        "Records",
        "NZ_SMDB",
        "Spectra_flatfiles",
        "NZdatabase_flatfile_FAS_horizontal_GeoMean.csv",
    ]
)

CMT_name = "GeoNet_CMT_solutions.csv"

sub = "/".join(cwd.split("/")[:-2] + ["geospatial", "Subduction_surfaces"])


########################################################################################################################
# FUNCTION DEFINITIONS
########################################################################################################################


def filter_tectclass(event_df, tectclass_df, cmt_tectclass_df):
    cmt_tectclass_df.rename(
        columns={"PublicID": "evid", "tectclass": "tect_class"}, inplace=True
    )
    cmt_tectclass_df["tect_method"] = "manual"

    tectclass_df["tect_class"] = tectclass_df["NGASUB_TectClass_Merged"]
    tectclass_df["tect_method"] = tectclass_df["NGASUB_Faults_Merged"]
    tectclass_df.loc[
        tectclass_df.NZSMDB_TectClass.isnull() == False, "tect_class"
    ] = tectclass_df[
        tectclass_df.NZSMDB_TectClass.isnull() == False
    ].NZSMDB_TectClass.values
    tectclass_df.loc[
        tectclass_df.NZSMDB_TectClass.isnull() == False, "tect_method"
    ] = "NZSMDB"
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


def get_domains(df, shapes, transformer):
    from shapely.geometry import Point  # conda install shapely
    from shapely.geometry.polygon import Polygon
    import numpy as np

    x, y = transformer.transform(df.lon, df.lat)
    points = [x, y]

    point = Point(points)

    no, name, type = [], [], []
    for layer in shapes:
        domain_no = layer["properties"]["Domain_No"]
        domain_name = layer["properties"]["DomainName"]
        domain_type = layer["properties"]["DomainType"]
        geometry_type = layer["geometry"]["type"]
        geometry_coords = layer["geometry"]["coordinates"]
        if geometry_type == "MultiPolygon":
            for coords in geometry_coords:
                polygon = Polygon(coords[0])
                if polygon.contains(point):
                    no, name, type = domain_no, domain_name, domain_type
        else:
            polygon = Polygon(geometry_coords[0])
            if polygon.contains(point):
                no, name, type = domain_no, domain_name, domain_type
    if not no:
        no, name, type = 0, "Oceanic", None
    return pd.Series([no, name, type])


def pos_longitude(lon):
    if lon < 0:
        lon = 360 + lon
    return lon


def merge_NZSMDB_flatfile_on_events(
    df,
    NZSMDB_csv_path=NZ_SMDB_path,
    left_on=False,
    right_on=False,
):
    """Merge metadata fields from NZ SMDB flatfile (Van Houtte, 2017)"""
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

    NZSMDB_df = pd.read_csv(NZSMDB_csv_path)
    NZSMDB_df.drop_duplicates(
        subset=["CuspID"],
        inplace=True,
    )
    new_name_dict = {col: "_".join(["NZSMDB", col]) for col in event_cols}
    NZSMDB_df = NZSMDB_df.rename(columns=new_name_dict)
    # print(NZSMDB_df.head())
    df = df.merge(
        right=NZSMDB_df[list(new_name_dict.values())],
        how="left",
        left_on=left_on,
        right_on=right_on,
    )

    return df


def merge_NZSMDB_flatfile_on_sites(
    df,
    NZSMDB_csv_path=NZ_SMDB_path,
    left_on=False,
    right_on=False,
):
    """Merge metadata fields from NZ SMDB flatfile (Van Houtte, 2017)"""
    site_cols = [
        "SiteCode",
        "SiteClass1170",
        "Vs30",
        "Vs30Uncert",
        "Tsite",
        "TsiteUncert",
        "Z1",
        "Z1Uncertainty",
    ]

    NZSMDB_df = pd.read_csv(NZSMDB_csv_path)
    NZSMDB_df.drop_duplicates(
        subset=["SiteCode"],
        inplace=True,
    )
    new_name_dict = {col: "_".join(["NZSMDB", col]) for col in site_cols}
    NZSMDB_df = NZSMDB_df.rename(columns=new_name_dict)
    # print(NZSMDB_df.head())
    df = df.merge(
        right=NZSMDB_df[list(new_name_dict.values())],
        how="left",
        left_on=left_on,
        right_on=right_on,
    )

    return df


def xyz_fault_points(
    fault_file,
    grid_space=7,
    R_a={},
    R_a_syn={},
    R_b={},
    R_c={},
    R_abc={},
    d_s=False,
    d_d=False,
    sep=",",
):
    """Determine an array of points on, and offshore, of each fault
    (PEER NGA SUB, 2020)

    Parameters
    ----------
    fault_file (str): Text file of longitude, latitude, and depth (-)
    grid_space (float): horizontal spacing of points to generate for R_a_syn
    R_a (dict): portion of fault in region a (lat,long,depth) NGA-SUB, 2020
    R_a_syn (dict): projected rectangle of region a (lat,long,depth) NGA-SUB, 2020
    R_b (dict): portion of fault in region b (lat,long,depth) NGA-SUB, 2020
    R_c (dict): portion of fault in region c (lat,long,depth) NGA-SUB, 2020
    R_abc (dict): total fault (lat,long,depth) NGA-SUB, 2020
    d_s (int,float): Upper limit of seismogenic zone, Hayes, 2018
    d_d (int,float): Lower limit of seismogenic zone, Hayes, 2018

    Returns
    -------
    R_a (dict): portion of faults in region a (lat,long,depth) NGA-SUB, 2020
    R_a_syn (dict): projected rectangle of region a for each fault (lat,long,depth) NGA-SUB, 2020
    R_b (dict): portion of faults in region b (lat,long,depth) NGA-SUB, 2020
    R_c (dict): portion of faults in region c (lat,long,depth) NGA-SUB, 2020
    R_abc (dict): total fault (lat,long,depth) NGA-SUB, 2020

    """

    # Read in the fault file
    #     df = pd.read_csv(
    #         fault_file,
    #         sep=sep,
    #         engine='python',
    #         )
    df = pd.read_csv(fault_file, sep=sep, engine="python", header=None)

    df.dropna(how="any", inplace=True)

    # Extract a fault name from the fault file
    fault = fault_file.split("/")[-1].replace(".txt", "").replace(".xyz", "")

    # Rename columns
    col_dict = {
        df.columns[-3]: "long",
        df.columns[-2]: "lat",
        df.columns[-1]: "depth",
    }

    df.rename(columns=col_dict, inplace=True)

    # Convert Z-Coord to positive depth value
    df["depth"] = df.apply(
        lambda x: np.abs(x["depth"]),
        axis=1,
    )

    # Convert all longitude values to positive (-179 becomes 181)
    df["long"] = df.apply(
        lambda x: pos_longitude(x["long"]),
        axis=1,
    )

    # Divide into regions
    df_a = df[(df.depth < d_s)]
    df_b = df[(df.depth >= d_s) & (df.depth <= d_d)]
    df_c = df[(df.depth > d_d)]

    # Add fault to fault surface dictionary
    R_a[fault] = df_a.to_numpy()
    R_b[fault] = df_b.to_numpy()
    R_c[fault] = df_c.to_numpy()
    R_abc[fault] = df.to_numpy()
    # print('np.shape(fault_surf[fault]): ',np.shape(fault_surf[fault]))

    # Project additional default offshore region to add to R_a_syn dictionary
    # Get centre of fault surface definition
    pt_1 = df["long"].idxmax(axis=0)
    pt_2 = df["long"].idxmin(axis=0)
    pt_3 = df["lat"].idxmax(axis=0)
    pt_4 = df["lat"].idxmin(axis=0)

    lonc_surface, latc_surface = geo.ll_mid(
        geo.ll_mid(
            df.long[pt_1],
            df.lat[pt_1],
            df.long[pt_2],
            df.lat[pt_2],
        )[0],
        geo.ll_mid(
            df.long[pt_1],
            df.lat[pt_1],
            df.long[pt_2],
            df.lat[pt_2],
        )[1],
        geo.ll_mid(
            df.long[pt_3],
            df.lat[pt_3],
            df.long[pt_4],
            df.lat[pt_4],
        )[0],
        geo.ll_mid(
            df.long[pt_3],
            df.lat[pt_3],
            df.long[pt_4],
            df.lat[pt_4],
        )[1],
    )

    # print('pt_1, pt_2, pt_3, pt_4: ', pt_1, pt_2, pt_3, pt_4)

    # Isolate an approximate updip edge at d_s
    df_updip_edge = df[(round(df.depth, 1) == d_s)]

    # Pick opposite ends of the updip edge
    pt_a = df_updip_edge["long"].idxmax(axis=0)
    pt_b = df_updip_edge["long"].idxmin(axis=0)

    # print('pt_a: ', pt_a)

    # Get centre of the approximate updip edge
    lonc_updip, latc_updip = geo.ll_mid(
        df_updip_edge.long[pt_a],
        df_updip_edge.lat[pt_a],
        df_updip_edge.long[pt_b],
        df_updip_edge.lat[pt_b],
    )
    # print('lonc_updip, latc_updip: ',lonc_updip, latc_updip)

    # Get distance from centre of updip edge to furtherest fault point
    # For computing the offshore distance
    corner_distances = [
        geo.ll_dist(
            df.long[pt_1],
            df.lat[pt_1],
            lonc_updip,
            latc_updip,
        ),
        geo.ll_dist(
            df.long[pt_2],
            df.lat[pt_2],
            lonc_updip,
            latc_updip,
        ),
        geo.ll_dist(
            df.long[pt_3],
            df.lat[pt_3],
            lonc_updip,
            latc_updip,
        ),
        geo.ll_dist(
            df.long[pt_4],
            df.lat[pt_4],
            lonc_updip,
            latc_updip,
        ),
    ]

    # Take the offshore distance as the average distance to the two furthest corners
    # of the fault surface from the center of the updip strike
    os_dist = np.mean(sorted(corner_distances, reverse=True)[:2])
    # print('os_dist: ', os_dist)

    # Determine strike bearing (arbitrary asimuth)
    strike_bearing = geo.ll_bearing(
        df_updip_edge.long[pt_a],
        df_updip_edge.lat[pt_a],
        df_updip_edge.long[pt_b],
        df_updip_edge.lat[pt_b],
    )
    # print('strike_bearing: ',strike_bearing)

    # Determine length along strike at updip edge
    strike_length = geo.ll_dist(
        df_updip_edge.long[pt_a],
        df_updip_edge.lat[pt_a],
        df_updip_edge.long[pt_b],
        df_updip_edge.lat[pt_b],
    )
    # print('strike_length: ',strike_length)

    # Determine the bearing
    updip_center_bearing = geo.ll_bearing(
        lonc_updip,
        latc_updip,
        lonc_surface,
        latc_surface,
    )

    # print('updip_center_bearing: ',updip_center_bearing)

    upper_limit = divmod(updip_center_bearing + 90 + 360, 360)[1]
    lower_limit = divmod(updip_center_bearing - 90 + 360, 360)[1]
    # print('upper_limit: ',upper_limit)
    # print('lower_limit: ',lower_limit)

    # Set offshore bearing as normal to strike bearing
    # and in opposite hemisphere as updip-center bearing
    for pot_offshore_bearing in [strike_bearing - 90, strike_bearing + 90]:
        if pot_offshore_bearing > upper_limit or pot_offshore_bearing < lower_limit:
            offshore_bearing = pot_offshore_bearing

    # print('offshore_bearing: ', offshore_bearing)

    # Points in offshore direction
    os_vect = np.linspace(
        0,
        os_dist,
        int(np.ceil(os_dist / grid_space)),
    )

    # Points in along-strike direction
    y_vect = np.linspace(
        -1 * strike_length / 2,
        strike_length / 2,
        int(np.ceil(strike_length / grid_space)),
    )

    amat_os, ainv_os = geo.gen_mat(offshore_bearing, lonc_updip, latc_updip)
    osy_arr = np.array([]).reshape(0, 2)

    # Create list of positions on each fault segment
    for os in os_vect:
        for y in y_vect:
            osy_arr = np.concatenate(
                (osy_arr, np.array([[os, y]])),
                axis=0,
            )

    ll_arr = geo.xy2ll(osy_arr, amat_os)
    depth_arr = np.zeros((ll_arr.shape[0], 1))

    R_a_syn[fault] = np.concatenate(
        (ll_arr, depth_arr),
        axis=1,
    )

    return R_a, R_a_syn, R_b, R_c, R_abc


def nhm_fault_points(
    faults,
    grid_space=3.5,
    fault_proj={},
    offshore_proj={},
    fault_surf={},
):
    """Determine an array of points on, and offshore, of each fault
    (PEER NGA SUB, 2020)

    Parameters
    ----------
    faults (dict): NHMFault generated by qcore.nhm.load_nhm()
    grid_space (float): horizontal spacing of points to generate
    fault_proj (dict): dictionary of arrays (lat,long)
    offshore_proj (dict): dictionary of offshore (updip) arrays (lat,long)
    fault_surf (dict): dictionary of arrays (x,y,z)

    Returns
    -------
    fault_surf (dict): appended dictionary of offshore (updip) arrays (long, lat, depth)
    offshore_proj (dict): appended dictionary of arrays (long,lat)
    """

    for k, fault in faults.items():
        z_dist = fault.dbottom - fault.dtop
        x_dist = np.abs(z_dist / np.tan(np.radians(fault.dip)))
        os_dist = x_dist

        # print('z_dist : ', z_dist)
        # print('x_dist : ', x_dist)
        # print('os_dist : ', os_dist)

        fault_ll_arr = np.array([]).reshape(0, 2)
        offshore_ll_arr = np.array([]).reshape(0, 2)

        for i, (lon1, lat1) in enumerate(fault.trace[:-1]):
            lon2, lat2 = fault.trace[
                i + 1
            ]  # arbitrary corner along top updip fault edge

            # Determine non-arbitrary bearing along updip fault edge
            strike = geo.ll_bearing(
                lon1,
                lat1,
                lon2,
                lat2,
            )

            # Determine center of updip fault edge
            lonc, latc = geo.ll_mid(
                lon1,
                lat1,
                lon2,
                lat2,
            )

            # Determine along-strike distance on updip fault edge
            y_dist = geo.ll_dist(
                lon1,
                lat1,
                lon2,
                lat2,
            )

            # Determine horizontal projection of points down dip
            x_t_vect = np.linspace(
                0,
                x_dist,
                int(np.ceil(x_dist / grid_space)),
            )

            # Determine horizontal projection of offshore points
            os_t_vect = np.linspace(
                0,
                os_dist,
                int(np.ceil(os_dist / grid_space)),
            )

            # Determine points along strike
            y_t_vect = np.linspace(
                -1 * y_dist / 2,
                y_dist / 2,
                int(np.ceil(y_dist / grid_space)),
            )

            # Determine vertical projection of points down dip
            z_vect = np.linspace(
                fault.dtop,
                fault.dbottom,
                int(np.ceil(x_dist / grid_space)),
            )

            # Generate fault model orientation
            amat, ainv = geo.gen_mat(
                strike,
                lonc,
                latc,
            )

            # Generate offshore model orientation
            amat_os, ainv_os = geo.gen_mat(
                strike + 180,
                lonc,
                latc,
            )

            xy_arr = np.array([]).reshape(0, 2)
            osy_arr = np.array([]).reshape(0, 2)
            xyz_arr = np.array([]).reshape(0, 3)

            theta = fault.dip_dir - strike

            # Create list of positions on each fault segment
            for x_t, z, os_t in zip(x_t_vect, z_vect, os_t_vect):
                for y_t in y_t_vect:
                    x = x_t * np.cos(np.radians(90 - theta))
                    y = y_t + x_t * np.sin(np.radians(90 - theta))
                    os = os_t * np.cos(np.radians(theta - 90))

                    xy_arr = np.concatenate((xy_arr, np.array([[x, y]])), axis=0)
                    osy_arr = np.concatenate((osy_arr, np.array([[os, y]])), axis=0)
                    xyz_arr = np.concatenate((xyz_arr, np.array([[x, y, z]])), axis=0)

            fault_ll_arr = geo.xy2ll(xy_arr, amat)
            offshore_ll_arr = geo.xy2ll(osy_arr, amat_os)

            fault_surf["_".join([fault.name, str(i)])] = np.concatenate(
                (fault_ll_arr[:, 0:], xyz_arr[:, 2:]), axis=1
            )
            offshore_proj["_".join([fault.name, str(i)])] = offshore_ll_arr

    return fault_surf, offshore_proj


def ngasub2020_tectclass(
    lat,
    lon,
    depth,
    fltsrf,
    offprj,
    fault_label=np.nan,
):
    """Applies the modified classification logic from the NGA-SUB 2020 report
    (PEER NGA SUB, 2020)

    Region A (vertical prism offshore of fault plane):
    depth<60km: 'Outer rise'
    depth>=60km: 'Slab'

    Region B (vertical prism containing fault plane):
    depth<min(shallowest interpretation of fault, 20km): 'Crustal'
    min(shallowest interpretation of fault, 20km)>depth>60km: 'Interface'
    depth>60km: 'Slab'

    Region C (everywhere else):
    depth<40km: 'Crustal'
    depth>=40km: 'Slab'


    Parameters
    ----------
    lat (float): latitude of hypocentre (degrees)
    long (float): longitude of hypocentre (degrees)
    depth (float): depth of hypocentre (km)
    fltsrf (dict): dictionary of arrays (long,lat,depth)
    offprj (dict): dictionary of offshore (updip) arrays (long,lat)


    Returns
    -------
    tectclass (str): 'Slab', 'Interface', 'Outer rise', or 'Crustal'
    fault (str): fault which triggered 'Interface' or 'Outer rise' tectclass labels
    """

    h_thresh = 5  # Horizontal distance extended tolerance for acceptance
    v_thresh = 5  # Vertical distance above point for acceptance

    # Iniatially classify as if farfield, correct later if neccessary
    if depth < 30:
        tectclass = "Crustal"
    else:
        tectclass = "Slab"

    if depth < 60:
        for flag, proj in zip(["offshore", "fault"], [offprj, fltsrf]):
            for k, arr in proj.items():
                i, d = geo.closest_location(arr[:, :2], lon, lat)
                if d < h_thresh:
                    fault_label = k
                    # Classifications for vertical prism offshore of fault plane
                    if flag == "offshore":
                        tectclass = "Outer rise"

                    # Classifications for vertical prism of fault plane
                    elif flag == "fault":
                        if depth < fltsrf[k][i][-1] - v_thresh:
                            tectclass = "Crustal"
                        else:
                            tectclass = "Interface"

    return tectclass, fault_label


def ngasub2020_tectclass_v2(
    lat,
    lon,
    depth,
    R_a=False,
    R_a_syn=False,
    R_b=False,
    R_c=False,
    fault_label=np.nan,
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
    lat (float): latitude of hypocentre (degrees)
    long (float): longitude of hypocentre (degrees)
    depth (float): depth of hypocentre (km)
    R_a (dict): portion of faults in region a (lat,long,depth) NGA-SUB, 2020
    R_a_syn (dict): projected rectangle of region a for each fault (lat,long,depth) NGA-SUB, 2020
    R_b (dict): portion of faults in region b (lat,long,depth) NGA-SUB, 2020
    R_c (dict): portion of faults in region c (lat,long,depth) NGA-SUB, 2020


    Returns
    -------
    tectclass (str): 'Slab', 'Interface', 'Outer rise', 'Crustal', or 'Undetermined'
    fault (str): fault which triggered 'Interface','Slab' or 'Outer rise' tectclass labels
    """

    h_thresh = 10  # Horizontal distance extended tolerance for acceptance
    v_thresh = 5  # Vertical distance above point for acceptance
    lon = pos_longitude(lon)

    # Initially classify as if farfield, correct later if neccessary
    if depth <= 30:
        tectclass = "Crustal"
    else:
        tectclass = "Undetermined"

    # for flag, region in zip(['A','A','C','B'],[R_a,R_a_syn,R_c,R_b]):
    for flag, region in zip(["A", "C", "B"], [R_a, R_c, R_b]):
        for fault, pt_arr in region.items():
            i, d = geo.closest_location(pt_arr[:, :2], lon, lat)
            if d < h_thresh:
                fault_label = fault
                # Classifications for region A
                if flag == "A":
                    if depth <= 60:
                        # tectclass = 'Outer rise'
                        tectclass = "Outer-rise"
                    else:
                        tectclass = "Slab"

                # Classifications for region B
                elif flag == "B":
                    if depth <= region[fault][i][-1] - 0.2 * depth and depth <= 20:
                        tectclass = "Crustal"
                    elif depth <= 60 and depth <= region[fault][i][-1] + 0.2 * depth:
                        tectclass = "Interface"
                    elif depth <= 60:
                        # tectclass = 'Slab (NGA Interface)'
                        tectclass = "Slab-(NGA-Interface)"
                    else:
                        tectclass = "Slab"

                # Classifications for region C
                elif flag == "C":
                    if depth <= 30:
                        tectclass = "Crustal"
                    elif depth >= region[fault][i][-1] - 0.2 * depth:
                        tectclass = "Slab"
                    else:
                        tectclass = "Undetermined"

    return tectclass, fault_label


def ngasub2020_tectclass_v3(
    lat,
    lon,
    depth,
    R_a=False,
    R_a_syn=False,
    R_b=False,
    R_c=False,
    fault_label=np.nan,
    h_thresh=10,
    v_thresh=10,
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
    lat (float): latitude of hypocentre (degrees)
    long (float): longitude of hypocentre (degrees)
    depth (float): depth of hypocentre (km)
    R_a (dict): portion of faults in region a (lat,long,depth) NGA-SUB, 2020
    R_a_syn (dict): projected rectangle of region a for each fault (lat,long,depth) NGA-SUB, 2020
    R_b (dict): portion of faults in region b (lat,long,depth) NGA-SUB, 2020
    R_c (dict): portion of faults in region c (lat,long,depth) NGA-SUB, 2020


    Returns
    -------
    tectclass (str): 'Slab', 'Interface', 'Outer rise', 'Crustal', or 'Undetermined'
    fault (str): fault which triggered 'Interface','Slab' or 'Outer rise' tectclass labels
    """

    # h_thresh = 10 # Horizontal distance extended tolerance for acceptance
    # v_thresh = 10 # Vertical distance above point for acceptance

    # Initially classify as if farfield, correct later if neccessary
    fault_depth = 0
    if depth <= 30:
        tectclass = "Crustal"
    elif depth > 60:
        tectclass = "Slab"
    else:
        tectclass = "Undetermined"

    # for flag, region in zip(['A','A','C','B'],[R_a,R_a_syn,R_c,R_b]):
    for flag, region in zip(["A", "C", "B"], [R_a, R_c, R_b]):
        for fault, pt_arr in region.items():
            i, d = geo.closest_location(pt_arr[:, :2], lon, lat)

            if d < h_thresh:
                fault_label = fault
                fault_depth = region[fault][i][-1]
                # Classifications for region A
                if flag == "A":
                    if depth <= 60:
                        # tectclass = 'Outer rise'
                        tectclass = "Outer-rise"
                    else:
                        tectclass = "Slab"

                # Classifications for region B
                elif flag == "B":
                    if depth <= region[fault][i][-1] - v_thresh and depth <= 20:
                        tectclass = "Crustal"
                    elif depth <= 60 and depth <= region[fault][i][-1] + v_thresh:
                        tectclass = "Interface"
                    # elif depth <= 60:
                    #     # tectclass = 'Slab (NGA Interface)'
                    #     tectclass = 'Slab-(NGA-Interface)'
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

    return tectclass, fault_label, fault_depth


###############################################################################
# SCRIPT RUNNING
###############################################################################

if __name__ == "__main__":
    # Tectonic classifications from GeoNet CMTs
    cmt_tectclass_df = pd.read_csv(
        "../focal/GeoNet-v04-tectclass.csv", low_memory=False
    )

    # Shape file for determining neotectonic domain
    shape = fiona.open("../TectonicDomains/TectonicDomains_Feb2021_8_NZTM.shp")

    directory = "../"
    geonet_cmt_df = pd.read_csv(
        directory + "focal/GeoNet_CMT_solutions.csv", low_memory=False
    )
    df = pd.read_csv(directory + "output/all_events.csv", low_memory=False)

    geonet_cmt_df.rename(columns={"PublicID": "evid"}, inplace=True)

    # Merge GEONET CMT data with original event database
    df = (
        df.set_index("evid")
        .join(
            geonet_cmt_df[["evid", "Mw", "Latitude", "Longitude", "CD"]].set_index(
                "evid"
            ),
            how="left",
            rsuffix="_CMT",
        )
        .reset_index()
    )
    df.loc[~df.Mw.isnull(), ["mag", "lat", "lon", "depth"]] = df.loc[
        ~df.Mw.isnull(), ["Mw", "Latitude", "Longitude", "CD"]
    ].values
    df.loc[~df.Mw.isnull(), ["mag_type", "mag_method", "loc_type", "loc_grid"]] = (
        "Mw",
        "CMT",
        "CMT",
        "CMT",
    )
    df.drop(columns=["Mw", "Latitude", "Longitude", "CD"], inplace=True)
    df["evid"] = df.evid.astype("object")

    #     df = pd.read_csv(
    #         '/Volumes/SeaJade 2 Backup/NZ/NZ_EQ_Catalog/testaroo/for_elena.csv',
    #         low_memory=False,
    #     )

    event_df = df.copy()

    # Determine the tectonic class for each record source based on hypo location
    df = merge_NZSMDB_flatfile_on_events(
        df.copy(),
        left_on="evid",
        right_on="NZSMDB_CuspID",
        # left_on='A_Record',
        # right_on='NZSMDB_Record',
    )

    #     df = merge_NZSMDB_flatfile_on_sites(
    #         df.copy(),
    #         left_on='sta',
    #         right_on='NZSMDB_SiteCode',
    #         )

    # Merged dataset for Kermadec and Hikurangi
    merged_a, merged_a_syn, merged_b, merged_c, merged_abc = xyz_fault_points(
        fault_file=sub + "/Merged_slab/hik_kerm_fault_300km_wgs84_poslon.txt",
        sep=",",
        d_s=10,  # Hayes et al., 2018
        d_d=47,  # Hayes et al., 2018
    )

    # Hayes et al., 2018
    merged_a, merged_a_syn, merged_b, merged_c, merged_abc = xyz_fault_points(
        fault_file=sub + "/Slab2_2018/puy/puy_slab2_dep_02.26.18.xyz",
        sep=",",
        d_s=11,  # Hayes et al., 2018
        d_d=30,  # Hayes et al., 2018
        R_a=merged_a,
        R_a_syn=merged_a_syn,
        R_b=merged_b,
        R_c=merged_c,
        R_abc=merged_abc,
    )

    pandarallel.initialize(
        nb_workers=8, progress_bar=True
    )  # Customize the number of parallel workers

    df["NGASUB_TectClass_Merged"] = df.parallel_apply(
        lambda x: ngasub2020_tectclass_v3(
            x["lat"],
            x["lon"],
            x["depth"],
            R_a=merged_a,
            R_a_syn=merged_a_syn,
            R_b=merged_b,
            R_c=merged_c,
        )[0],
        axis=1,
    )

    df["NGASUB_Faults_Merged"] = df.parallel_apply(
        lambda x: ngasub2020_tectclass_v3(
            x["lat"],
            x["lon"],
            x["depth"],
            R_a=merged_a,
            R_a_syn=merged_a_syn,
            R_b=merged_b,
            R_c=merged_c,
        )[1],
        axis=1,
    )

    # Merge tectonic classification data from both CMT and regular event data
    merged_df = filter_tectclass(event_df, df, cmt_tectclass_df)

    # Determine tectonic domain
    shapes = []
    for layer in shape:
        domain_no = layer["properties"]["Domain_No"]
        domain_name = layer["properties"]["DomainName"]
        print(domain_no, domain_name)
        shapes.append(layer)
    wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)
    pandarallel.initialize(
        nb_workers=8, progress_bar=True
    )  # Customize the number of parallel workers
    merged_df[["domain_no", "domain_name", "domain_type"]] = merged_df.parallel_apply(
        lambda x: get_domains(x, shapes, wgs2nztm), axis=1
    )
    merged_df.drop(columns="domain_name", inplace=True)

    # Hikurangi and Puysegur
    # Williams, 2012
    #     HP_a, HP_a_syn, HP_b, HP_c, HP_abc = xyz_fault_points(
    #         fault_file=sub+'/Hik_Williams_2012/new_charles_low_res.txt',
    #         sep='\s+',
    #         d_s=10, # Hayes et al., 2018
    #         d_d=47, # Hayes et al., 2018
    #         )
    #
    # #     Hayes et al., 2018
    #     HP_a, HP_a_syn, HP_b, HP_c, HP_abc = xyz_fault_points(
    #         fault_file=sub+'/Slab2_2018/puy/puy_slab2_dep_02.26.18.xyz',
    #         sep=',',
    #         d_s=11, # Hayes et al., 2018
    #         d_d=30, # Hayes et al., 2018
    #         R_a = HP_a,
    #         R_a_syn = HP_a_syn,
    #         R_b = HP_b,
    #         R_c = HP_c,
    #         R_abc = HP_abc,
    #         )
    #
    #     pandarallel.initialize(nb_workers=6) # Customize the number of parallel workers
    #
    #     df['NGASUB_TectClass_HikPuy'] = df.parallel_apply(
    #         lambda x: ngasub2020_tectclass_v3(
    #             x['lat'],
    #             x['lon'],
    #             x['depth'],
    #             R_a = HP_a,
    #             R_a_syn = HP_a_syn,
    #             R_b = HP_b,
    #             R_c = HP_c,
    #             )[0],
    #         axis=1)
    #
    #     df['NGASUB_Faults_HikPuy'] = df.parallel_apply(
    #         lambda x: ngasub2020_tectclass_v3(
    #             x['lat'],
    #             x['lon'],
    #             x['depth'],
    #             R_a = HP_a,
    #             R_a_syn = HP_a_syn,
    #             R_b = HP_b,
    #             R_c = HP_c,
    #             )[1],
    #         axis=1)
    #
    #     # Kermadec and Pusegur
    #     # Hayes et al., 2018
    #     KP_a, KP_a_syn, KP_b, KP_c, KP_abc = xyz_fault_points(
    #         fault_file=sub+'/Slab2_2018/ker/ker_slab2_dep_02.24.18.xyz',
    #         sep=',',
    #         d_s=10, # Hayes et al., 2018
    #         d_d=47, # Hayes et al., 2018
    #         )
    #
    #     # Hayes et al., 2018
    #     KP_a, KP_a_syn, KP_b, KP_c, KP_abc = xyz_fault_points(
    #         fault_file=sub+'/Slab2_2018/puy/puy_slab2_dep_02.26.18.xyz',
    #         sep=',',
    #         d_s=11, # Hayes et al., 2018
    #         d_d=30, # Hayes et al., 2018
    #         R_a = KP_a,
    #         R_a_syn = KP_a_syn,
    #         R_b = KP_b,
    #         R_c = KP_c,
    #         R_abc = KP_abc,
    #         )
    #
    #     pandarallel.initialize(nb_workers=6) # Customize the number of parallel workers
    #
    #     df['NGASUB_TectClass_KerPuy'] = df.parallel_apply(
    #         lambda x: ngasub2020_tectclass_v3(
    #             x['lat'],
    #             x['lon'],
    #             x['depth'],
    #             R_a = KP_a,
    #             R_a_syn = KP_a_syn,
    #             R_b = KP_b,
    #             R_c = KP_c,
    #             )[0],
    #         axis=1)
    #
    #     df['NGASUB_Faults_KerPuy'] = df.parallel_apply(
    #         lambda x: ngasub2020_tectclass_v3(
    #             x['lat'],
    #             x['lon'],
    #             x['depth'],
    #             R_a = KP_a,
    #             R_a_syn = KP_a_syn,
    #             R_b = KP_b,
    #             R_c = KP_c,
    #             )[1],
    #         axis=1)

    merged_df.to_csv(
        out_path + "/earthquake_source_table_tectdomain.csv",
        mode="w",
        index=False,
    )
