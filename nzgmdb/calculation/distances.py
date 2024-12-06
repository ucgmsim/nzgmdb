import functools
import multiprocessing as mp
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client as FDSN_Client
from pyproj import Transformer
from shapely.geometry import Point
from shapely.geometry.polygon import LineString, Polygon
from source_modelling import srf

from empirical.util import estimations
from nzgmdb.CCLD import ccldpy
from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from qcore import coordinates, geo, grid, src_site_dist


def calc_fnorm_slip(
    strike: float, dip: float, rake: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the normal and slip vectors from strike, dip and rake
    Parameters
    ----------
    strike : float
        The strike angle of the fault in degrees
    dip : float
        The dip angle of the fault in degrees
    rake : float
        The rake angle of the fault in degrees

    Returns
    -------
    fnorm : np.ndarray
        The normal vector of the fault
    slip : np.ndarray
        The slip vector of the fault
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


def get_domain_focal(
    domain_no: int, domain_focal_df: pd.DataFrame
) -> tuple[int, int, int]:
    """
    Get the focal mechanism for a given domain
    If not found, return the default values
    Strike: 220
    Rake: 45
    Dip: 90

    Parameters
    ----------
    domain_no : int
        The domain number
    domain_focal_df : pd.DataFrame
        The domain focal data

    Returns
    -------
    strike : int
        The strike angle of the fault in degrees
    rake : int
        The rake angle of the fault in degrees
    dip : int
        The dip angle of the fault in degrees
    """
    if domain_no == 0:
        return 220, 45, 90
    else:
        domain = domain_focal_df[domain_focal_df.Domain_No == domain_no].iloc[0]
        return domain.strike, domain.rake, domain.dip


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
):
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
    dict
        A dictionary containing the following keys:
        'strike' : float
            The strike angle of the fault in degrees
        'dip' : float
            The dip angle of the fault in degrees
        'rake' : float
            The rake angle of the fault in degrees
        'ztor' : float
            The depth to the top of the rupture in km
        'dbottom' : float
            The depth to the bottom of the rupture in km
        'length' : float
            The length of the fault along strike in km
        'dip_dist' : float
            The width of the fault down dip in km
        'hyp_lat' : float
            The latitude of the hypocenter
        'hyp_lon' : float
            The longitude of the hypocenter
        'hyp_strike' : float
            The hypocenter along-strike position (0 - 1)
        'hyp_dip' : float
            The hypocenter down-dip position (0 - 1)
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
        "hyp_strike": selected["Hypocenter Along-Strike Position"].values[0],
        "hyp_dip": selected["Hypocenter Down-Dip Position"].values[0],
    }


def get_nodal_plane_info(
    event_id: str,
    event_row: pd.Series,
    geonet_cmt_df: pd.DataFrame,
    modified_cmt_df: pd.DataFrame,
    domain_focal_df: pd.DataFrame,
    srf_files: dict,
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
    geonet_cmt_df : pd.DataFrame
        The Geonet CMT data
    modified_cmt_df : pd.DataFrame
        The modified CMT data for the correct nodal plane
    domain_focal_df : pd.DataFrame
        The focal mechanism data for the different domains
    srf_files : dict
        The srf files for specific events

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

    # Check if the event_id is in the srf_files
    if event_id in srf_files:
        # Read the srf file to determine the nodal plane information
        srf_model = srf.read_srf(srf_files[event_id])
        nodal_plane_info["f_type"] = "ff"

        # Find the total slip and average rake for each subfault
        total_slip = [
            np.sum(plane_points["slip"]) for plane_points in srf_model.segments
        ]
        avg_rake = [
            np.average(plane_points["rake"]) for plane_points in srf_model.segments
        ]

        # Calculate the average strike, dip and rake based on weighted average of slip
        (
            nodal_plane_info["strike"],
            nodal_plane_info["dip"],
            nodal_plane_info["rake"],
        ) = estimations.calculate_avg_strike_dip_rake(
            srf_model.planes, avg_rake, total_slip
        )

        config = cfg.Config()
        points_per_km = config.get_value("points_per_km")

        srf_points = []
        for plane in srf_model.planes:
            corner_0, corner_1, corner_2, _ = plane.corners
            # Utilise grid functions from qcore to get the mesh grid
            plane_points = grid.coordinate_meshgrid(
                corner_0, corner_1, corner_2, 1000 / points_per_km
            )
            # Reshape to (n, 3)
            plane_points = plane_points.reshape(-1, 3)
            srf_points.append(plane_points)
        srf_points = np.vstack(srf_points)
        # Swap the lat and lon for the srf points
        nodal_plane_info["srf_points"] = srf_points[:, [1, 0, 2]]

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
        nodal_plane_info["dip_dist"] = sum([plane.width for plane in srf_model.planes])

        # Check if there is only 1 plane
        if len(srf_model.planes) == 1:
            plane = srf_model.planes[0]
            nodal_plane_info["corner_0"] = plane.corners[0]
            nodal_plane_info["corner_1"] = plane.corners[1]
            nodal_plane_info["corner_2"] = plane.corners[2]
            nodal_plane_info["corner_3"] = plane.corners[3]
        else:
            # Ensure the corners are None
            nodal_plane_info["corner_0"] = [None, None, None]
            nodal_plane_info["corner_1"] = [None, None, None]
            nodal_plane_info["corner_2"] = [None, None, None]
            nodal_plane_info["corner_3"] = [None, None, None]

    elif event_id in modified_cmt_df.PublicID.values:
        # Event is in the modified CMT data
        nodal_plane_info["f_type"] = "cmt"
        cmt = modified_cmt_df[modified_cmt_df.PublicID == event_id].iloc[0]
        # Compute the CCLD Simulations for the event
        ccld_info = run_ccld_simulation(
            event_id, event_row, cmt.strike1, cmt.dip1, cmt.rake1, "A"
        )

    elif event_id in geonet_cmt_df.PublicID.values:
        # Event is in the Geonet CMT data
        # Need to determine the correct plane
        nodal_plane_info["f_type"] = "cmt_unc"
        cmt = geonet_cmt_df[geonet_cmt_df.PublicID == event_id].iloc[0]

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
        nodal_plane_info["f_type"] = "domain"
        strike, rake, dip = get_domain_focal(event_row["domain_no"], domain_focal_df)

        # Compute the CCLD Simulations for the event
        ccld_info = run_ccld_simulation(event_id, event_row, strike, dip, rake, "D")

    if ccld_info is not None:
        # Update the nodal plane info with the ccld info
        nodal_plane_info.update(ccld_info)

    return nodal_plane_info


def compute_distances_for_event(
    event_row: pd.Series,
    im_df: pd.DataFrame,
    station_df: pd.DataFrame,
    modified_cmt_df: pd.DataFrame,
    geonet_cmt_df: pd.DataFrame,
    domain_focal_df: pd.DataFrame,
    taupo_polygon: Polygon,
    srf_files: dict,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Compute the distances for a given event

    Parameters
    ----------
    event_row : pd.Series
        The event row from the earthquake source table
    im_df : pd.DataFrame
        The full IM data from the catalog
    station_df : pd.DataFrame
        The full station data
    modified_cmt_df : pd.DataFrame
        The modified CMT data for the correct nodal plane
    geonet_cmt_df : pd.DataFrame
        The Geonet CMT data
    domain_focal_df : pd.DataFrame
        The focal mechanism data for the different domains
    taupo_polygon : Polygon
        The Taupo VZ polygon
    srf_files : dict
        The srf files for specific events

    Returns
    -------
    propagation_data_combo : pd.DataFrame
        The propagation data for the event
    extra_event_data : pd.DataFrame
        The extra event data for the event which includes the correct nodal plane information
    """

    # Extract out the relevant event_row data
    event_id = event_row["evid"]
    im_event_df = im_df[im_df["evid"] == event_id]

    # Check if the event doesn't have IM data
    # If it doesn't, skip the event
    if im_event_df.empty:
        return None, None

    # Get the station data
    event_sta_df = station_df[station_df["sta"].isin(im_event_df["sta"])].reset_index()
    stations = event_sta_df[["lon", "lat", "depth"]].to_numpy()

    # Get the nodal plane information
    nodal_plane_info = get_nodal_plane_info(
        event_id,
        event_row,
        geonet_cmt_df,
        modified_cmt_df,
        domain_focal_df,
        srf_files,
    )
    (
        strike,
        rake,
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
        hyp_strike,
        hyp_dip,
        corner_0,
        corner_1,
        corner_2,
        corner_3,
    ) = (
        nodal_plane_info["strike"],
        nodal_plane_info["rake"],
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
        nodal_plane_info["hyp_strike"],
        nodal_plane_info["hyp_dip"],
        nodal_plane_info["corner_0"],
        nodal_plane_info["corner_1"],
        nodal_plane_info["corner_2"],
        nodal_plane_info["corner_3"],
    )

    if srf_header is None or srf_points is None:
        # Calculate the corners of the plane
        dip_dir = (strike + 90) % 360
        projected_width = dip_dist * np.cos(np.radians(dip))

        config = cfg.Config()
        points_per_km = config.get_value("points_per_km")

        # Find the center of the plane based on the hypocentre location
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
    rrups_lat, rrups_lon = rrup_points[:, 0], rrup_points[:, 1]

    r_epis = geo.get_distances(
        np.dstack([event_sta_df.lon.values, event_sta_df.lat.values])[0],
        event_row["lon"],
        event_row["lat"],
    )
    r_hyps = np.sqrt(r_epis**2 + (event_row["depth"] - event_sta_df.depth.values) ** 2)
    azs = np.array(
        [
            geo.ll_bearing(event_row["lon"], event_row["lat"], station[0], station[1])
            for station in stations
        ]
    )
    b_azs = np.array(
        [
            geo.ll_bearing(station[0], station[1], event_row["lon"], event_row["lat"])
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
                        "net": station.net,
                        "sta": station.sta,
                        "r_epi": r_epis[station_index],
                        "r_hyp": r_hyps[station_index],
                        "r_jb": rjbs[station_index],
                        "r_rup": rrups[station_index],
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
                "hyp_lat": hyp_lat,
                "hyp_lon": hyp_lon,
                "hyp_strike": hyp_strike,
                "hyp_dip": hyp_dip,
                "corner_0_lat": corner_0[0],
                "corner_0_lon": corner_0[1],
                "corner_0_depth": corner_0[2],
                "corner_1_lat": corner_1[0],
                "corner_1_lon": corner_1[1],
                "corner_1_depth": corner_1[2],
                "corner_2_lat": corner_2[0],
                "corner_2_lon": corner_2[1],
                "corner_2_depth": corner_2[2],
                "corner_3_lat": corner_3[0],
                "corner_3_lon": corner_3[1],
                "corner_3_depth": corner_3[2],
            },
        ]
    )

    return propagation_data_combo, extra_event_data


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
    # Get the data / flatfile directory
    data_dir = file_structure.get_data_dir()
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Get the modified CMT data
    modified_cmt_file = (
        data_dir / "GeoNet_CMT_solutions_20201129_PreferredNodalPlane_v1.csv"
    )
    modified_cmt_df = pd.read_csv(modified_cmt_file)

    # Get the regular CMT data
    config = cfg.Config()
    geonet_cmt_df = pd.read_csv(config.get_value("cmt_url"), low_memory=False)

    # Load the eq source table
    event_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_TECTONIC,
        dtype={"evid": str},
    )

    # Get the focal domain
    domain_focal_df = pd.read_csv(
        data_dir / "focal_mech_tectonic_domain_v1.csv", low_memory=False
    )

    # Get the Taupo VZ polygon
    tect_domain_points = pd.read_csv(
        data_dir / "tect_domain" / "tectonic_domain_polygon_points.csv",
        low_memory=False,
    )
    tvz_points = tect_domain_points[tect_domain_points.domain_no == 4][
        ["latitude", "longitude"]
    ]
    ll_num = config.get_value("ll_num")
    nztm_num = config.get_value("nztm_num")
    wgs2nztm = Transformer.from_crs(ll_num, nztm_num)
    taupo_transform = np.dstack(
        np.array(wgs2nztm.transform(tvz_points.latitude, tvz_points.longitude))
    )[0]
    taupo_polygon = Polygon(taupo_transform)

    # Check the SrfSourceModels directory for the srf files
    if not (data_dir / "SrfSourceModels").exists():
        # Check for the SrfSourceModels.zip and unzip it
        srf_zip = data_dir / "SrfSourceModels.zip"
        if srf_zip.exists():
            with zipfile.ZipFile(srf_zip, "r") as zip_ref:
                # Extract all the contents to the destination directory
                zip_ref.extractall(data_dir)
        else:
            raise FileNotFoundError(
                "The SrfSourceModels directory and zip file does not exist"
            )

    # Get the srf files as a dictionary of file paths and event_id as the key
    srf_files = {}
    for srf_file in (data_dir / "SrfSourceModels").glob("*.srf"):
        srf_files[srf_file.stem] = srf_file

    # Get the IM data
    im_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.GROUND_MOTION_IM_CATALOGUE,
        dtype={"evid": str},
    )

    # Get the station information
    client_NZ = FDSN_Client("GEONET")
    inventory = client_NZ.get_stations()
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
    station_df = pd.DataFrame(
        station_info, columns=["net", "sta", "lat", "lon", "elev"]
    )
    station_df = station_df.drop_duplicates().reset_index(drop=True)

    # Select unique stations from IM data and merge
    im_station_df = im_df[["sta"]].drop_duplicates()
    station_df = pd.merge(im_station_df, station_df, on="sta", how="left")
    station_df["depth"] = station_df["elev"] / -1000

    with mp.Pool(n_procs) as p:
        result_dfs = p.map(
            functools.partial(
                compute_distances_for_event,
                im_df=im_df,
                station_df=station_df,
                modified_cmt_df=modified_cmt_df,
                geonet_cmt_df=geonet_cmt_df,
                domain_focal_df=domain_focal_df,
                taupo_polygon=taupo_polygon,
                srf_files=srf_files,
            ),
            [row for idx, row in event_df.iterrows()],
        )

    # Combine the results
    propagation_results, extra_event_results = zip(*result_dfs)
    propagation_data = pd.concat(propagation_results)
    extra_event_data = pd.concat(extra_event_results)

    # Merge the extra event data with the event data
    event_df = pd.merge(event_df, extra_event_data, on="evid", how="right")

    # Save the results
    propagation_data.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.PROPAGATION_TABLE, index=False
    )
    event_df.to_csv(
        flatfile_dir
        / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_DISTANCES,
        index=False,
    )
