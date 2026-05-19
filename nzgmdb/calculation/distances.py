"""
This module contains functions to calculate distances between earthquake planes and sites, as well
as determining the rupture plane geometry for a given event.
"""

import functools
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Optional, TypedDict

import fiona
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon

from cmt_solutions import cmt_data
from nzgmdb.CCLD import ccldpy
from nzgmdb.data_retrieval import tect_domain
from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from nzgmdb.management.data_registry import NZGMDB_DATA, REGISTRY
from oq_wrapper import estimations
from qcore import coordinates, geo, grid, src_site_dist
from source_modelling import magnitude_scaling, srf


class FocalMechanism(TypedDict):
    """
    Represents the geometric and spatial parameters of a crustal domain
    focal mechanism.
    """

    strike: float
    """The strike angle of the fault in degrees."""
    dip: float
    """The dip angle of the fault in degrees."""
    rake: float
    """The rake angle of the fault in degrees."""
    ztor: float
    """The depth to the top of the rupture in km."""
    dbottom: float
    """The depth to the bottom of the rupture in km."""
    length: float
    """The length of the fault along strike in km."""
    dip_dist: float
    """The width of the fault down dip in km."""
    hyp_lat: float
    """The latitude of the hypocentre."""
    hyp_lon: float
    """The longitude of the hypocentre."""
    hyp_depth: float
    """The depth of the hypocentre in km."""
    hyp_strike: float
    """The hypocentre along-strike position (0 - 1)."""
    hyp_dip: float
    """The hypocentre down-dip position (0 - 1)."""


def calc_fnorm_slip(
    strike: float, dip: float, rake: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the normal and slip vectors from strike, dip, and rake angles.

    Parameters
    ----------
    strike : float
        The strike angle of the fault in degrees.
    dip : float
        The dip angle of the fault in degrees.
    rake : float
        The rake angle of the fault in degrees.

    Returns
    -------
    fnorm : np.ndarray
        The normal vector of the fault.
    slip : np.ndarray
        The slip vector of the fault.
    """
    phi = np.deg2rad(strike)
    delt = np.deg2rad(dip)
    lam = np.deg2rad(rake)

    fnorm = np.asarray(
        [-np.sin(delt) * np.sin(phi), np.sin(delt) * np.cos(phi), -np.cos(delt)]
    )
    slip = np.asarray(
        [
            np.cos(lam) * np.cos(phi) + np.cos(delt) * np.sin(lam) * np.sin(phi),
            np.cos(lam) * np.sin(phi) - np.cos(delt) * np.sin(lam) * np.cos(phi),
            -np.sin(lam) * np.sin(delt),
        ]
    )

    return fnorm, slip


def run_ccld_simulation(
    event_id: str,
    event_row: pd.Series,
    strike: float,
    dip: float,
    rake: float,
    method: str,
    strike2: float = None,
    dip2: float = None,
    rake2: float = None,
) -> FocalMechanism:
    """
    Run the CCLD simulation for an event.
    Uses default values for the number of simulations based on the tectonic class mentioned
    in the CCLDpy Documentation.

    Parameters
    ----------
    event_id : str
        The event id
    event_row : pd.Series
        The event row from the earthquake source table (Must contain the following columns: lat, lon, depth, mag, tect_class)
    strike : float
        The strike angle of the fault in degrees for the first plane
    dip : float
        The dip angle of the fault in degrees for the first plane
    rake : float
        The rake angle of the fault in degrees for the first plane
    method : str
        The method to use for the simulation (A, B, C, D or E)
    strike2 : float, optional
        The strike angle of the fault in degrees of a potential second plane, by default None
    dip2 : float, optional
        The dip angle of the fault in degrees of a potential second plane, by default None
    rake2 : float, optional
        The rake angle of the fault in degrees of a potential second plane, by default None

    Returns
    -------
    FocalMechanism
        A dictionary containing the calculated fault geometry and hypocentre parameters.
    """
    ccdl_tect_class = ccldpy.TECTONIC_MAPPING[event_row.tect_class]
    # Extra check for undetermined tectonic class
    if event_row.tect_class == "Undetermined":
        config = cfg.Config()
        # Check if the depth is greater than 50km and if so set it to slab
        ccdl_tect_class = (
            "crustal"
            if event_row.depth <= config.get_value("crustal_depth")
            else "intraslab"
        )
    if ccdl_tect_class == "crustal":
        nsims = [334, 333, 333, 111, 111, 111, 0]
    elif ccdl_tect_class == "intraslab":
        nsims = [0, 0, 0, 0, 0, 0, 333]
    else:
        # Interface
        nsims = [0, 0, 333, 0, 0, 0, 333]
    _, selected = ccldpy.simulate_rupture_surface(
        int(event_id.split("p")[-1]),
        ccdl_tect_class,
        "other",
        event_row.lat,
        event_row.lon,
        event_row.depth,
        event_row.mag,
        method,
        nsims,
        strike=strike,
        dip=dip,
        rake=rake,
        strike2=strike2,
        dip2=dip2,
        rake2=rake2,
    )

    return {
        "strike": selected["Strike"].values[0],
        "dip": selected["Dip"].values[0],
        "rake": selected["Rake"].values[0],
        "ztor": selected["Rupture Top Depth (km)"].values[0],
        "dbottom": selected["Rupture Bottom Depth (km)"].values[0],
        "length": selected["Length (km)"].values[0],
        "dip_dist": selected["Width (km)"].values[0],
        "hyp_lat": selected["Hypocenter Latitude"].values[0],
        "hyp_lon": selected["Hypocenter Longitude"].values[0],
        "hyp_depth": selected["Hypocenter Depth (km)"].values[0],
        "hyp_strike": selected["Hypocenter Along-Strike Position"].values[0],
        "hyp_dip": selected["Hypocenter Down-Dip Position"].values[0],
    }


def get_crustal_domain_focal(
    event_id: str,
    event_row: pd.Series,
    nz_mech: dict,
    length_bin: str,
    domain_no_backup: int,
    domain_focal_df: pd.DataFrame,
) -> FocalMechanism:
    """
    Select the appropriate nodal plane from the Crustal domain focal mechanism data.
    If both cases are the same, select the highest probability and run CCLD simulations for that plane.
    If the cases are different, select the highest probability from each case and run CCLD simulations for both planes.
    If the domain number is not found, use the backup focal mechanism.

    Parameters
    ----------
    event_id : str
        The event id
    event_row : pd.Series
        The event row from the earthquake source table
    nz_mech : dict
        The domain focal mechanism data
    length_bin : str
        The length bin to use for the focal mechanism selection
    domain_no_backup : int
        The domain number for the backup focal mechanism.
    domain_focal_df : pd.DataFrame
        The focal mechanism data for the different domains.

    Returns
    -------
    FocalMechanism
        A dictionary containing the calculated fault geometry and hypocentre parameters.
    """
    try:
        domain_model = nz_mech[event_row["domain_no"]]
    except KeyError:
        # Use the backup focal mechanism
        strike, dip, rake = get_backup_focal_mechanism(
            domain_no_backup, domain_focal_df
        )
        return run_ccld_simulation(event_id, event_row, strike, dip, rake, "D")

    case1 = domain_model["case1"][length_bin]
    case2 = domain_model["case2"][length_bin]

    # Check if the cases are the same
    cases_equal = (
        case1["strikeAn"] == case2["strikeAn"]
        and case1["dipAn"] == case2["dipAn"]
        and case1["rakeAn"] == case2["rakeAn"]
        and case1["prob"] == case2["prob"]
    )

    # Select the highest probability for case 1
    idx_max = int(np.argmax(case1["prob"]))
    strike = float(case1["strikeAn"][idx_max])
    dip = float(case1["dipAn"][idx_max])
    rake = float(case1["rakeAn"][idx_max])

    if cases_equal:
        # Compute the CCLD Simulations for the event
        ccld_info = run_ccld_simulation(event_id, event_row, strike, dip, rake, "D")
    else:
        # Select the highest probability for case 2
        case2_idx_max = int(np.argmax(case2["prob"]))
        strike2 = float(case2["strikeAn"][case2_idx_max])
        dip2 = float(case2["dipAn"][case2_idx_max])
        rake2 = float(case2["rakeAn"][case2_idx_max])
        # Compute the CCLD Simulations for the event with both possible planes
        ccld_info = run_ccld_simulation(
            event_id, event_row, strike, dip, rake, "C", strike2, dip2, rake2
        )
    return ccld_info


def get_backup_focal_mechanism(
    domain_no_backup: int, domain_focal_df: pd.DataFrame
) -> tuple[float, float, float]:
    """
    Retrieves the backup focal mechanism.

    Parameters
    ----------
    domain_no_backup : int
        The domain number for the backup focal mechanism.
    domain_focal_df : pd.DataFrame
        The focal mechanism data for the different domains.

    Returns
    -------
    tuple
        A tuple containing the strike, rake, and dip angles.
    """
    if domain_no_backup == 0:
        return 220, 45, 90
    domain = domain_focal_df[domain_focal_df.Domain_No == domain_no_backup].iloc[0]
    return domain.strike, domain.rake, domain.dip


def get_nodal_plane_info(
    event_id: str,
    event_row: pd.Series,
    cmt_df: pd.DataFrame,
    domain_focal_df: pd.DataFrame,
    srf_files: dict,
    hik_objs: np.ndarray,
    puy_objs: np.ndarray,
    nz_mech: dict,
    slab_faulting_geo: dict,
) -> dict:
    """
    Determine the correct nodal plane for the event
    First checks if the event is in the srf_files, if it is, it uses the srf file to determine the nodal plane
    If it is not in the srf_files, it checks if the event is in the rupture_models to determine the nodal plane
    If it is not in the modified CMT data, it checks if the event is in the Geonet CMT data to determine the nodal plane
    If it is not in the Geonet CMT data, it uses the domain focal to determine the nodal plane

    Extra variables such as the length, dip_dist, srf_points, srf_header, ztor, dbottom are also determined in
    some of these scenarios

    Parameters
    ----------
    event_id : str
        The event id
    event_row : pd.Series
        The event row from the earthquake source table
    cmt_df : pd.DataFrame
        The Centroid Moment Tensor data for New Zealand events
    domain_focal_df : pd.DataFrame
        The focal mechanism data for the different domains as a backup
    srf_files : dict
        The srf files for specific events
    hik_objs : np.ndarray
        The Hikurangi RBF objects for strike and dip interpolation as well as the Hikurangi footprint for Crustal events
    puy_objs : np.ndarray
        The Puysegur RBF objects for strike and dip interpolation as well as the Puysegur footprint for Crustal events
    nz_mech : dict
        The domain focal mechanism data for each domain
    slab_faulting_geo : dict
        The slab faulting geometry data for both Hikurangi and Puysegur

    Returns
    -------
    dict
        A dictionary containing the following keys:
        'strike' : float
            The strike angle of the fault in degrees
        'rake' : float
            The rake angle of the fault in degrees
        'dip' : float
            The dip angle of the fault in degrees
        'ztor' : float
            The depth to the top of the rupture in km
        'dbottom' : float
            The depth to the bottom of the rupture in km
        'length' : float
            The length of the fault along strike in km
        'dip_dist' : float
            The width of the fault down dip in km
        'srf_points' : np.ndarray
            The points of the fault planes
        'srf_header' : list
            The header of the fault planes
        'f_type' : str
            The focal type that determined the nodal plane (ff, geonet_rm, cmt, cmt_unc, domain)
    """
    # Create the default return to be filled using defaultdict
    nodal_plane_info = defaultdict(lambda: None)
    ccld_info = None

    # Split the cmt data into reviewed and unreviewed data
    reviewed_cmt_data = cmt_df[cmt_df["reviewed"]]
    unreviewed_cmt_data = cmt_df[~cmt_df["reviewed"]]

    # Check if the event_id is in the srf_files
    if event_id in srf_files:
        # Read the srf file to determine the nodal plane information
        srf_model = srf.read_srf(srf_files[event_id])
        nodal_plane_info["f_type"] = "ff"

        # Find the plane areas and average rake for each subfault
        plane_areas = [plane.length * plane.width for plane in srf_model.planes]
        avg_rake = [
            np.average(plane_points["rake"]) for plane_points in srf_model.segments
        ]
        nodal_plane_info["avg_rake"] = avg_rake

        # Calculate the average strike, dip and rake based on weighted average of slip
        (
            nodal_plane_info["strike"],
            nodal_plane_info["dip"],
            nodal_plane_info["rake"],
            nodal_plane_info["dip_dist"],
        ) = estimations.calculate_avg_multi_plane_properties(
            srf_model.planes, avg_rake, plane_areas
        )

        nodal_plane_info["srf_points"] = srf_model.points.loc[
            :, ["lon", "lat", "dep"]
        ].to_numpy()

        # Generate the srf header
        nodal_plane_info["srf_header"] = (
            srf_model.header[["nstk", "ndip", "stk", "len", "wid"]]
            .rename(
                columns={
                    "nstk": "nstrike",
                    "ndip": "ndip",
                    "stk": "strike",
                    "len": "length",
                    "wid": "width",
                }
            )
            .to_dict(orient="records")
        )

        nodal_plane_info["ztor"] = min(
            [plane.top_m / 1000 for plane in srf_model.planes]
        )
        nodal_plane_info["dbottom"] = max(
            [plane.bottom_m / 1000 for plane in srf_model.planes]
        )
        nodal_plane_info["length"] = sum([plane.length for plane in srf_model.planes])

        nodal_plane_info["planes"] = srf_model.planes

        # Find the location of the hypocentre in the srf model
        # Grab the point at which tinit is 0
        hyp_point = srf_model.points[srf_model.points["tinit"] == 0]
        nodal_plane_info["hyp_lat"] = hyp_point["lat"].values[0]
        nodal_plane_info["hyp_lon"] = hyp_point["lon"].values[0]
        nodal_plane_info["hyp_depth"] = hyp_point["dep"].values[0]

        # Grab the header information and get the nstk * ndip for each plane
        nstk_ndip = [
            header.nstk * header.ndip for ix, header in srf_model.header.iterrows()
        ]
        # Cumulate the nstk * ndip to find the plane index
        cum_nstk_ndip = np.cumsum(nstk_ndip)
        # Use the index of the hyp_point in the srf_points to find the plane index
        hyp_index = hyp_point.index.values[0]
        plane_index = np.searchsorted(cum_nstk_ndip, hyp_index, side="right")

        # Now we can find the s_hyp and d_hyp
        nstk = srf_model.header.iloc[plane_index]["nstk"]
        ndip = srf_model.header.iloc[plane_index]["ndip"]

        plane_start_index = 0 if plane_index == 0 else cum_nstk_ndip[plane_index - 1]
        local_index = hyp_index - plane_start_index

        dip_idx = local_index // nstk
        stk_idx = local_index % nstk

        nodal_plane_info["hyp_strike"] = stk_idx / (nstk - 1)
        nodal_plane_info["hyp_dip"] = dip_idx / (ndip - 1)
        nodal_plane_info["plane_index"] = plane_index

    elif event_id in reviewed_cmt_data.PublicID.values:
        # Event is in the reviewed CMT data
        nodal_plane_info["f_type"] = "cmt"
        cmt = reviewed_cmt_data[reviewed_cmt_data.PublicID == event_id].iloc[0]
        # Compute the CCLD Simulations for the event
        ccld_info = run_ccld_simulation(
            event_id, event_row, cmt.strike1, cmt.dip1, cmt.rake1, "A"
        )

    elif event_id in unreviewed_cmt_data.PublicID.values:
        # Event is in the Geonet CMT data, however it has not been reviewed
        nodal_plane_info["f_type"] = "cmt_unc"
        cmt = unreviewed_cmt_data[unreviewed_cmt_data.PublicID == event_id].iloc[0]

        # Compute the CCLD Simulations for the event
        ccld_info = run_ccld_simulation(
            event_id,
            event_row,
            cmt.strike1,
            cmt.dip1,
            cmt.rake1,
            "C",
            cmt.strike2,
            cmt.dip2,
            cmt.rake2,
        )
    else:
        # Event is not found in any of the datasets
        # Use the domain focal
        hik_strike_rbf, hik_dip_rbf, hik_footprint = hik_objs
        puy_strike_rbf, puy_dip_rbf, puy_footprint = puy_objs
        domain_no_backup = event_row["domain_no_backup"]
        nodal_plane_info["f_type"] = "domain"

        if event_row["tect_class"] == "Crustal":
            # First assume strike-slip to estimate length
            length = magnitude_scaling.leonard_magnitude_to_length(event_row.mag, 15)
            length_bin = ">45" if length > 45.0 else ">15"

            ccld_info = get_crustal_domain_focal(
                event_id,
                event_row,
                nz_mech,
                length_bin,
                domain_no_backup,
                domain_focal_df,
            )

            # Check the new length to see if a different length bin should be used
            new_length = ccld_info["length"]
            new_length_bin = ">45" if new_length > 45.0 else ">15"
            if new_length_bin != length_bin:
                # Recompute with the new length bin
                ccld_info = get_crustal_domain_focal(
                    event_id,
                    event_row,
                    nz_mech,
                    new_length_bin,
                    domain_no_backup,
                    domain_focal_df,
                )
        elif event_row["tect_class"] == "Interface":
            lat, lon = event_row["lat"], event_row["lon"]

            rake = None
            # Check which subduction zone the event is in
            if hik_footprint.contains(Point(lon, lat)):
                strike = float(np.squeeze(hik_strike_rbf([[lon, lat]])))
                dip = float(np.squeeze(hik_dip_rbf([[lon, lat]])))
            elif puy_footprint.contains(Point(lon, lat)):
                strike = float(np.squeeze(puy_strike_rbf([[lon, lat]])))
                dip = float(np.squeeze(puy_dip_rbf([[lon, lat]])))
            else:
                strike, rake, dip = get_backup_focal_mechanism(
                    domain_no_backup, domain_focal_df
                )
            # Check for infinite values
            if not np.isfinite(strike) or not np.isfinite(dip):
                strike, rake, dip = get_backup_focal_mechanism(
                    domain_no_backup, domain_focal_df
                )
            rake = 90.0 if rake is None else rake

            # Run ccld to get length, width, ztor, dbottom
            ccld_info = run_ccld_simulation(event_id, event_row, strike, dip, rake, "D")

        elif event_row["tect_class"] == "Slab":
            lat, lon = event_row["lat"], event_row["lon"]
            # Check which zone the event is in
            if hik_footprint.contains(Point(lon, lat)):
                tbl = slab_faulting_geo["hik"]
            elif puy_footprint.contains(Point(lon, lat)):
                tbl = slab_faulting_geo["puy"]
            else:
                strike, rake, dip = get_backup_focal_mechanism(
                    domain_no_backup, domain_focal_df
                )
                # Run ccld to get length, width, ztor, dbottom
                ccld_info = run_ccld_simulation(
                    event_id, event_row, strike, dip, rake, "D"
                )
                nodal_plane_info.update(ccld_info)
                return nodal_plane_info

            # Find the closest point in the table
            depth_bins = [int(b) for b in tbl.keys()]
            bin = np.array(
                tbl[
                    str(
                        depth_bins[
                            np.argmin(np.abs(np.array(depth_bins) - event_row["depth"]))
                        ]
                    )
                ]
            )
            # Select the highest probability (doesn't matter which case as they are the same)
            idx_max = int(np.argmax(bin[:, 3]))
            strike = float(bin[idx_max, 0]) % 360
            dip = float(bin[idx_max, 1])
            rake = float(bin[idx_max, 2])

            # Run ccld to get length, width, ztor, dbottom
            ccld_info = run_ccld_simulation(event_id, event_row, strike, dip, rake, "D")

        else:
            strike, rake, dip = get_backup_focal_mechanism(
                domain_no_backup, domain_focal_df
            )
            ccld_info = run_ccld_simulation(event_id, event_row, strike, dip, rake, "D")

    if ccld_info is not None:
        # Update the nodal plane info with the ccld info
        nodal_plane_info.update(ccld_info)

    return nodal_plane_info


def compute_distances_for_event(
    event_row: pd.Series,
    im_df: pd.DataFrame,
    site_df: pd.DataFrame,
    cmt_df: pd.DataFrame,
    domain_focal_df: pd.DataFrame,
    taupo_polygon: Polygon,
    srf_files: dict,
    hik_objs: np.ndarray,
    puy_objs: np.ndarray,
    nz_mech: dict,
    slab_faulting_geo: dict,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compute the distances for a given event

    Parameters
    ----------
    event_row : pd.Series
        The event row from the earthquake source table
    im_df : pd.DataFrame
        The full IM data from the catalogue
    site_df : pd.DataFrame
        The full site data
    cmt_df : pd.DataFrame
        The Centroid Moment Tensor data
    domain_focal_df : pd.DataFrame
        The focal mechanism data for the different domains
    taupo_polygon : Polygon
        The Taupo VZ polygon
    srf_files : dict
        The srf files for specific events
    hik_objs : np.ndarray
        The Hikurangi RBF objects for strike and dip interpolation as well as the Hikurangi footprint for Crustal events
    puy_objs : np.ndarray
        The Puysegur RBF objects for strike and dip interpolation as well as the Puysegur footprint for Crustal events
    nz_mech : dict
        The domain focal mechanism data for each domain
    slab_faulting_geo : dict
        The slab faulting geometry data for both Hikurangi and Puysegur

    Returns
    -------
    propagation_data_combo : pd.DataFrame
        The propagation data for the event
    extra_event_data : pd.DataFrame
        The extra event data for the event which includes the correct nodal plane information
    geometry_data : pd.DataFrame
        The geometry data for the event which includes the corners of the rupture plane / planes
    """

    # Extract out the relevant event_row data
    event_id = event_row["evid"]
    im_event_df = im_df[im_df["evid"] == event_id]

    # Check if the event doesn't have IM data
    # If it doesn't, skip the event
    if im_event_df.empty:
        return None, None, None

    # Get the site data
    event_sta_df = site_df[site_df["sta"].isin(im_event_df["sta"])].reset_index()
    stations = event_sta_df[["lon", "lat", "depth"]].to_numpy()

    # Get the nodal plane information
    nodal_plane_info = get_nodal_plane_info(
        event_id,
        event_row,
        cmt_df,
        domain_focal_df,
        srf_files,
        hik_objs,
        puy_objs,
        nz_mech,
        slab_faulting_geo,
    )
    (
        strike,
        rake,
        avg_rake,
        dip,
        length,
        dip_dist,
        srf_points,
        srf_header,
        ztor,
        dbottom,
        f_type,
        hyp_lat,
        hyp_lon,
        hyp_depth,
        hyp_strike,
        hyp_dip,
        hyp_plane_index,
        corner_0,
        corner_1,
        corner_2,
        corner_3,
        planes,
    ) = (
        nodal_plane_info["strike"],
        nodal_plane_info["rake"],
        nodal_plane_info["avg_rake"],
        nodal_plane_info["dip"],
        nodal_plane_info["length"],
        nodal_plane_info["dip_dist"],
        nodal_plane_info["srf_points"],
        nodal_plane_info["srf_header"],
        nodal_plane_info["ztor"],
        nodal_plane_info["dbottom"],
        nodal_plane_info["f_type"],
        nodal_plane_info["hyp_lat"],
        nodal_plane_info["hyp_lon"],
        nodal_plane_info["hyp_depth"],
        nodal_plane_info["hyp_strike"],
        nodal_plane_info["hyp_dip"],
        nodal_plane_info["plane_index"],
        nodal_plane_info["corner_0"],
        nodal_plane_info["corner_1"],
        nodal_plane_info["corner_2"],
        nodal_plane_info["corner_3"],
        nodal_plane_info["planes"],
    )

    if srf_header is None or srf_points is None:
        # Calculate the corners of the plane
        dip_dir = (strike + 90) % 360
        projected_width = dip_dist * np.cos(np.radians(dip))

        config = cfg.Config()
        points_per_km = config.get_value("points_per_km")

        # Find the centre of the plane based on the hypocentre location
        strike_direction = np.array(
            [np.cos(np.radians(strike)), np.sin(np.radians(strike))]
        )
        dip_direction = np.array(
            [np.cos(np.radians(dip_dir)), np.sin(np.radians(dip_dir))]
        )

        # Convert the hypocentre location to NZTM
        hyp_nztm = coordinates.wgs_depth_to_nztm(np.asarray([hyp_lat, hyp_lon]))

        # Calculate the distance needed to travel in the strike direction
        strike_centroid_dist = length * 1000 / 2
        strike_hyp_dist = hyp_strike * length * 1000
        strike_diff_dist = strike_centroid_dist - strike_hyp_dist

        # Calculate the distance needed to travel in the dip direction
        dip_centroid_dist = projected_width * 1000 / 2
        dip_hyp_dist = hyp_dip * projected_width * 1000
        dip_diff_dist = dip_centroid_dist - dip_hyp_dist

        # Calculate the centre of the plane
        centroid = hyp_nztm + np.array([strike_diff_dist, dip_diff_dist]) @ np.array(
            [strike_direction, dip_direction]
        )

        # Convert back to lat, lon
        centroid_lat_lon = coordinates.nztm_to_wgs_depth(centroid)

        # Get the corners of the srf points
        corner_0, corner_1, corner_2, corner_3 = grid.grid_corners(
            centroid_lat_lon,
            strike,
            dip_dir,
            ztor,
            dbottom,
            length,
            projected_width,
        )

        # Utilise grid functions from qcore to get the mesh grid
        srf_points = grid.coordinate_meshgrid(
            corner_0, corner_1, corner_2, 1000 / points_per_km
        )

        # Reshape to (n, 3)
        srf_points = srf_points.reshape(-1, 3)
        # Swap the lat and lon for the srf points
        srf_points = srf_points[:, [1, 0, 2]]

        # Generate the srf header
        nstrike = int(round(length * points_per_km))
        ndip = int(round(dip_dist * points_per_km))
        srf_header = [
            {
                "nstrike": nstrike,
                "ndip": ndip,
                "strike": strike,
                "length": length,
                "width": dip_dist,
            }
        ]

    # Divide the srf depth points by 1000 to convert to km
    srf_points[:, 2] /= 1000

    # Calculate the distances
    rrups, rjbs, rrup_points = src_site_dist.calc_rrup_rjb(
        srf_points, stations, return_rrup_points=True
    )
    rxs, rys = src_site_dist.calc_rx_ry(srf_points, srf_header, stations)
    rrups_lon, rrups_lat = rrup_points[:, 0], rrup_points[:, 1]

    # Get the segment corners for the srf or corners
    if event_id in srf_files:
        seg_corners = np.zeros((3, 4, len(planes)))
        for i, plane in enumerate(planes):
            for j, idx in enumerate([0, 1, 3, 2]):  # Ordering to match corner mapping
                seg_corners[:, j, i] = (
                    coordinates.wgs_depth_to_nztm(plane.corners[idx])[[1, 0, 2]]
                    / 1000.0
                )

    else:
        # If not in srf_files, use the nodal plane info corners
        seg_corners = np.zeros((3, 4, 1))
        for i, corner in enumerate(
            [corner_0, corner_1, corner_3, corner_2]
        ):  # Map to correct corner order
            seg_corners[:, i, 0] = (
                coordinates.wgs_depth_to_nztm(np.array(corner))[[1, 0, 2]] / 1000.0
            )

    # Flip the stations index 0 and 1 to match for NZTM convention
    nztm_stations = (
        coordinates.wgs_depth_to_nztm(stations[:, [1, 0, 2]])[:, [1, 0, 2]] / 1000
    )

    # Calculate Ravg
    ravgs = compute_ravg_distance_vectorized(seg_corners, nztm_stations)

    r_epis = geo.get_distances(
        np.dstack([event_sta_df.lon.values, event_sta_df.lat.values])[0],
        nodal_plane_info["hyp_lon"],
        nodal_plane_info["hyp_lat"],
    )
    r_hyps = np.sqrt(
        r_epis**2 + (nodal_plane_info["hyp_depth"] - event_sta_df.depth.values) ** 2
    )
    azs = np.array(
        [
            geo.ll_bearing(
                nodal_plane_info["hyp_lon"],
                nodal_plane_info["hyp_lat"],
                station[0],
                station[1],
            )
            for station in stations
        ]
    )
    b_azs = np.array(
        [
            geo.ll_bearing(
                station[0],
                station[1],
                nodal_plane_info["hyp_lon"],
                nodal_plane_info["hyp_lat"],
            )
            for station in stations
        ]
    )

    # Determine if the path goes through the Taupo VZ
    # and calculate the length of the path that goes through the Taupo VZ
    tvz_lengths, boundary_dists_rjb = distance_in_taupo(
        event_sta_df,
        taupo_polygon,
        rjbs,
        rrups_lon,
        rrups_lat,
    )

    # Create the propagation data per station
    propagation_data = []
    for station_index, station in event_sta_df.iterrows():
        propagation_data.append(
            pd.DataFrame(
                [
                    {
                        "evid": event_id,
                        "provider": station.provider,
                        "net": station.net,
                        "sta": station.sta,
                        "r_epi": r_epis[station_index],
                        "r_hyp": r_hyps[station_index],
                        "r_jb": rjbs[station_index],
                        "r_rup": rrups[station_index],
                        "r_avg": ravgs[station_index],
                        "r_x": rxs[station_index],
                        "r_y": rys[station_index],
                        "r_tvz": tvz_lengths[station_index],
                        "r_xvf": boundary_dists_rjb[station_index],
                        "az": azs[station_index],
                        "b_az": b_azs[station_index],
                        "f_type": f_type,
                        "reloc": event_row["reloc"],
                    },
                ]
            )
        )
    propagation_data_combo = pd.concat(propagation_data)

    # Create the extra event data
    extra_event_data = pd.DataFrame(
        [
            {
                "evid": event_id,
                "strike": strike,
                "dip": dip,
                "rake": rake,
                "f_length": length,
                "f_width": dip_dist,
                "f_type": f_type,
                "z_tor": ztor,
                "z_bor": dbottom,
            },
        ]
    )

    # Create the geometry data per plane
    if event_id in srf_files:
        geometry_rows = []
        for plane_id, plane in enumerate(planes, start=1):
            corners = plane.corners
            geometry_rows.append(
                {
                    "evid": event_id,
                    "plane_id": plane_id,
                    "f_type": f_type,
                    "strike": plane.strike,
                    "dip": plane.dip,
                    "rake": avg_rake[plane_id - 1],
                    "f_length": plane.length,
                    "f_width": plane.width,
                    "z_tor": plane.top_m / 1000,
                    "z_bor": plane.bottom_m / 1000,
                    "hyp_lat": hyp_lat,
                    "hyp_lon": hyp_lon,
                    "hyp_depth": hyp_depth,
                    "hyp_strike": (
                        hyp_strike if plane_id == hyp_plane_index + 1 else None
                    ),
                    "hyp_dip": hyp_dip if plane_id == hyp_plane_index + 1 else None,
                    "corner_0_lat": corners[0][0],
                    "corner_0_lon": corners[0][1],
                    "corner_0_depth": corners[0][2] / 1000.0,
                    "corner_1_lat": corners[1][0],
                    "corner_1_lon": corners[1][1],
                    "corner_1_depth": corners[1][2] / 1000.0,
                    "corner_2_lat": corners[2][0],
                    "corner_2_lon": corners[2][1],
                    "corner_2_depth": corners[2][2] / 1000.0,
                    "corner_3_lat": corners[3][0],
                    "corner_3_lon": corners[3][1],
                    "corner_3_depth": corners[3][2] / 1000.0,
                }
            )

        geometry_data = pd.DataFrame(geometry_rows)
    else:
        geometry_data = pd.DataFrame(
            [
                {
                    "evid": event_id,
                    "plane_id": 1,
                    "f_type": f_type,
                    "strike": strike,
                    "dip": dip,
                    "rake": rake,
                    "f_length": length,
                    "f_width": dip_dist,
                    "z_tor": ztor,
                    "z_bor": dbottom,
                    "hyp_lat": hyp_lat,
                    "hyp_lon": hyp_lon,
                    "hyp_depth": hyp_depth,
                    "hyp_strike": hyp_strike,
                    "hyp_dip": hyp_dip,
                    "corner_0_lat": corner_0[0],
                    "corner_0_lon": corner_0[1],
                    "corner_0_depth": corner_0[2] / 1000.0,
                    "corner_1_lat": corner_1[0],
                    "corner_1_lon": corner_1[1],
                    "corner_1_depth": corner_1[2] / 1000.0,
                    "corner_2_lat": corner_3[0],
                    "corner_2_lon": corner_3[1],
                    "corner_2_depth": corner_3[2] / 1000.0,
                    "corner_3_lat": corner_2[0],
                    "corner_3_lon": corner_2[1],
                    "corner_3_depth": corner_2[2] / 1000.0,
                }
            ]
        )

    return propagation_data_combo, extra_event_data, geometry_data


def perpendicular_height(
    point: np.ndarray, base_start: np.ndarray, base_end: np.ndarray
) -> float:
    """
    Compute perpendicular height from a point to a line defined by two points.

    Parameters
    ----------
    point : np.ndarray, shape (3,)
        The point from which the height is measured.
    base_start : np.ndarray, shape (3,)
        The start point of the line segment.
    base_end : np.ndarray, shape (3,)
        The end point of the line segment.

    Returns
    -------
    float
        The perpendicular height from the point to the line segment.
    """
    base_vec = base_end - base_start
    point_vec = point - base_start
    cross = np.cross(base_vec, point_vec)
    base_len = np.linalg.norm(base_vec)
    return np.linalg.norm(cross) / base_len if base_len else 0.0


def inverse_square_integral(
    sites: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    """
    Vectorised inverse square integral over a segment for multiple sites.

    Parameters
    ----------
    sites : np.ndarray, shape (n_sites, 3)
        The sites (x, y, z) coordinates.
    p1 : np.ndarray, shape (3,)
        The first point of the segment (x, y, z).
    p2 : np.ndarray, shape (3,)
        The second point of the segment (x, y, z).

    Returns
    -------
    np.ndarray, shape (n_sites,)
        The integral values for each site.
    """
    vec1 = p1 - sites
    vec2 = p2 - p1
    B = np.sum(vec1 * vec2, axis=1)
    C = np.dot(vec2, vec2)
    D = np.sum(np.cross(vec1, vec2) ** 2, axis=1)
    sqrt_D = np.sqrt(D)
    atan_diff = np.arctan2(C + B, sqrt_D) - np.arctan2(B, sqrt_D)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(D > 0, atan_diff / sqrt_D, 0.0)
    return result


def compute_ravg_distance_vectorized(
    seg_corners: np.ndarray, sites: np.ndarray
) -> np.ndarray:
    """
    Vectorised Ravg calculation for multiple site locations.

    Parameters
    ----------
    seg_corners : np.ndarray, shape (3, 4, n_segments)
        Fault plane corner segments
    sites : np.ndarray, shape (n_sites, 3)
        Site (x, y, z) coordinates

    Returns
    -------
    np.ndarray
        Ravg values for each site, shape (n_sites,)
    """
    vertical_step = 0.1
    n_sites = sites.shape[0]
    n_segments = seg_corners.shape[2]
    # Set the site depth to 0
    sites[:, 2] = 0.0
    sum_inv_rsq = np.zeros(n_sites)
    for i in range(n_segments):
        TL, TR, BR, BL = seg_corners[:, :, i].T
        height = perpendicular_height(TL, TR, BR)
        n_steps = int(np.ceil(height / vertical_step))
        left_deltas = (BL - TL) / n_steps
        right_deltas = (BR - TR) / n_steps
        seg_start = TL + left_deltas * 0.5
        seg_end = TR + right_deltas * 0.5
        for _ in range(n_steps):
            result = inverse_square_integral(sites, seg_start, seg_end)
            sum_inv_rsq += result / (n_steps * n_segments)
            seg_start += left_deltas
            seg_end += right_deltas
    return np.sqrt(1.0 / sum_inv_rsq)


def distance_in_taupo(
    sta_df: pd.DataFrame,
    taupo_polygon: Polygon,
    r_epis: np.ndarray,
    rrups_lon: np.ndarray,
    rrups_lat: np.ndarray,
) -> tuple[list, list]:
    """
    Figures out if the path from the station to the event goes through the Taupo VZ
    And if it does to calculate the length of the path that goes through the Taupo VZ

    Parameters
    ----------
    sta_df : pd.DataFrame
        The station data for a given event
    taupo_polygon : Polygon
        The Taupo VZ polygon
    r_epis : np.ndarray
        The epicentral distances for the stations
    rrups_lon : np.ndarray
        The longitude of the rupture points closest to the station
    rrups_lat : np.ndarray
        The latitude of the rupture points closest to the station

    Returns
    -------
    tvz_lengths : list
        The length of the path that goes through the Taupo VZ for each station
    boundary_dists_rjb : list
        The distance from the station to the boundary of the Taupo VZ polygon for each station
    """
    # Taupo VZ polygon acquired from https://www.geonet.org.nz/data/supplementary/earthquake_location_grope

    config = cfg.Config()
    ll_num = config.get_value("ll_num")
    nztm_num = config.get_value("nztm_num")
    wgs2nztm = Transformer.from_crs(ll_num, nztm_num)

    # Transform the rrups to NZTM
    rrups_transform = wgs2nztm.transform(rrups_lat, rrups_lon)

    tvz_lengths = []
    boundary_dists_rjb = []

    # Loop through all the stations
    for station_index, station in sta_df.iterrows():
        # Create the line between the station and the event
        sta_transform = wgs2nztm.transform(station.lat, station.lon)
        line = LineString(
            [
                [rrups_transform[0][station_index], rrups_transform[1][station_index]],
                [sta_transform[0], sta_transform[1]],
            ]
        )

        tvz_length = 0
        boundary_dist_rjb = None

        # Check if the line intersects the Taupo VZ polygon
        if line.intersection(taupo_polygon):
            # If it does, calculate the length of the line that goes through the Taupo VZ
            # Get the intersection point with the boundary
            point = taupo_polygon.boundary.intersection(line)

            if taupo_polygon.contains(Point(sta_transform)):
                # If the line is completely inside the Taupo VZ polygon
                boundary_dist_rjb = 0
            else:
                # If the line intersects the boundary of the Taupo VZ polygon
                if point.geom_type == "MultiPoint":
                    point = point.geoms[0]
                if point.geom_type != "LineString":
                    # Calculate the distance from the station to the boundary
                    boundary_dist_rjb = (
                        np.sqrt(
                            (point.x - sta_transform[0]) ** 2
                            + (point.y - sta_transform[1]) ** 2
                        )
                        / 1000
                    )

            line_points = line.intersection(taupo_polygon)
            tvz_length = min(line_points.length / 1000 / r_epis[station_index], 1)

        tvz_lengths.append(tvz_length)
        boundary_dists_rjb.append(boundary_dist_rjb)

    return tvz_lengths, boundary_dists_rjb


def calc_distances(main_dir: Path, n_procs: int = 1):
    """
    Calculate the distances for the propagation path table
    Also determines the fault plane for the event and merges it into the event table

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    n_procs : int
        The number of processes to use for the calculation (per event)
    """
    # Get the flatfile directory
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Get the CMT solutions data
    cmt_df = cmt_data.get_cmt_data()

    # Load the eq source table
    event_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_TECTONIC,
        dtype={"evid": str},
    )

    NZGMDB_DATA.fetch("TectonicDomains_Feb2021_8_NZTM.shp")
    NZGMDB_DATA.fetch("TectonicDomains_Feb2021_8_NZTM.dbf")
    NZGMDB_DATA.fetch("TectonicDomains_Feb2021_8_NZTM.shx")
    with fiona.open(
        Path(NZGMDB_DATA.abspath) / "TectonicDomains_Feb2021_8_NZTM.shp"
    ) as collection:
        shapes = list(collection)

    fallback_domain_values = tect_domain.find_domain_from_shapes(
        event_df.loc[:, ["lat", "lon"]],
        shapes,
    )

    # Add the fallback domain values to the event df
    event_df["domain_no_backup"] = fallback_domain_values.loc[:, "domain_no"].values

    hik_objs = np.load(NZGMDB_DATA.fetch("hik_focmec.npy"), allow_pickle=True)[()]
    puy_objs = np.load(NZGMDB_DATA.fetch("puy_focmec.npy"), allow_pickle=True)[()]

    with open(NZGMDB_DATA.fetch("slab-faulting2.json"), "r", encoding="utf-8") as f:
        slab_faulting_geo = json.load(f)

    with open(NZGMDB_DATA.fetch("nzfocmecmod.json"), "r", encoding="utf-8") as f:
        nz_mech = json.load(f)

    # Get the focal domain
    domain_focal_df = pd.read_csv(
        NZGMDB_DATA.fetch("focal_mech_tectonic_domain_v1.csv"),
    )

    # Get the Taupo VZ polygon
    tect_domain_points = pd.read_csv(
        NZGMDB_DATA.fetch("tectonic_domain_polygon_points.csv"),
    )
    tvz_points = tect_domain_points[tect_domain_points.domain_no == 4][
        ["latitude", "longitude"]
    ]
    config = cfg.Config()
    ll_num = config.get_value("ll_num")
    nztm_num = config.get_value("nztm_num")
    wgs2nztm = Transformer.from_crs(ll_num, nztm_num)
    taupo_transform = np.dstack(
        np.array(wgs2nztm.transform(tvz_points.latitude, tvz_points.longitude))
    )[0]
    taupo_polygon = Polygon(taupo_transform)

    # Go through the registry keys and check if they are .srf files to use
    srf_files = {}
    for file in REGISTRY.keys():
        if file.endswith(".srf"):
            # If they are, add them to the srf files dictionary and fetch them
            NZGMDB_DATA.fetch(file)
            srf_files[Path(file).stem] = Path(NZGMDB_DATA.abspath) / file

    # Get the IM data to know what stations to calculate the distances for each event
    im_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.GROUND_MOTION_IM_CATALOGUE,
        dtype={"evid": str},
        usecols=["evid", "sta"],
    )

    # Get the site information
    site_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.SITE_TABLE,
        dtype={"sta": str},
    )
    site_df = site_df.loc[:, ["sta", "provider", "net", "lat", "lon", "elev"]]

    # Select unique stations from IM data and merge
    im_station_df = im_df[["sta"]].drop_duplicates()
    site_df = pd.merge(im_station_df, site_df, on="sta", how="left")
    site_df["depth"] = site_df["elev"] / -1000

    with mp.Pool(n_procs) as p:
        result_dfs = p.map(
            functools.partial(
                compute_distances_for_event,
                im_df=im_df,
                site_df=site_df,
                cmt_df=cmt_df,
                domain_focal_df=domain_focal_df,
                taupo_polygon=taupo_polygon,
                srf_files=srf_files,
                hik_objs=hik_objs,
                puy_objs=puy_objs,
                nz_mech=nz_mech,
                slab_faulting_geo=slab_faulting_geo,
            ),
            [row for idx, row in event_df.iterrows()],
        )

    # Combine the results
    propagation_results, extra_event_results, geometry_results = zip(*result_dfs)
    propagation_data = pd.concat(propagation_results)
    extra_event_data = pd.concat(extra_event_results)
    geometry_data = pd.concat(geometry_results)

    # Merge the extra event data with the event data
    event_df = pd.merge(event_df, extra_event_data, on="evid", how="right")

    # Remove the domain_no_backup column
    event_df = event_df.drop(columns=["domain_no_backup"])

    # Save the results
    propagation_data.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.PROPAGATION_TABLE, index=False
    )
    geometry_data.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_GEOMETRY,
        index=False,
    )
    event_df.to_csv(
        flatfile_dir
        / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_DISTANCES,
        index=False,
    )
