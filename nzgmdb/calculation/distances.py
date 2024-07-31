from pathlib import Path
from typing import Optional
import functools
import multiprocessing as mp

import numpy as np
import pandas as pd
from pyproj import Transformer
from obspy.clients.fdsn import Client as FDSN_Client
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString

from qcore import srf, geo, grid
from qcore.uncertainties import mag_scaling
from qcore.uncertainties.magnitude_scaling import strasser_2010
from IM_calculation.source_site_dist import src_site_dist
from nzgmdb.data_retrieval import rupture_models as geonet_rupture_models
from nzgmdb.management import file_structure, config as cfg
from nzgmdb.CCLD import ccldpy


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


def mech_rot(
    norm1: np.ndarray, norm2: np.ndarray, slip1: np.ndarray, slip2: np.ndarray
) -> int:
    """
    Determine the correct nodal plane for the event by checking 4 different
    mechanisms and determining the correct plane based on the rotation that is as close to 0 as possible

    Parameters
    ----------
    norm1 : np.ndarray
        The normal vector of the first mechanism
    norm2 : np.ndarray
        The normal vector of the second mechanism
    slip1 : np.ndarray
        The slip vector of the first mechanism
    slip2 : np.ndarray
        The slip vector of the second mechanism

    Returns
    -------
    plane_out : int
        The plane that is closest to 0 rotation
    """
    b1 = np.cross(norm1, slip1)

    rotations = np.zeros(4)
    for iteration in range(0, 4):
        if iteration < 2:
            norm2_temp = norm2
            slip2_temp = slip2
        else:
            norm2_temp = slip2
            slip2_temp = norm2
        if iteration in {1, 3}:
            norm2_temp = tuple(-x for x in norm2_temp)
            slip2_temp = tuple(-x for x in slip2_temp)

        b2 = np.cross(norm2_temp, slip2_temp)

        phi1 = np.dot(norm1, norm2_temp)
        phi2 = np.dot(slip1, slip2_temp)
        phi3 = np.dot(b1, b2)

        # In some cases, identical dot products produce values incrementally higher than 1
        phi1 = np.arccos(np.clip(phi1, -1, 1))
        phi2 = np.arccos(np.clip(phi2, -1, 1))
        phi3 = np.arccos(np.clip(phi3, -1, 1))

        # Set array of phi for indexing reasons later
        phi = [phi1, phi2, phi3]

        # If the mechanisms are very close, rotation = 0
        epsilon = 1e-4
        if np.allclose(phi, 0, atol=epsilon):
            rotations[iteration] = 0
        # If one vector is the same, it is the rotation axis
        elif phi1 < epsilon:
            rotations[iteration] = np.rad2deg(phi2)
        elif phi2 < epsilon:
            rotations[iteration] = np.rad2deg(phi3)
        elif phi3 < epsilon:
            rotations[iteration] = np.rad2deg(phi1)

        # Find difference vectors - the rotation axis must be orthogonal to all three of
        # these vectors
        else:
            difference_vectors = np.array(
                [
                    np.array(norm1) - np.array(norm2_temp),
                    np.array(slip1) - np.array(slip2_temp),
                    np.array(b1) - np.array(b2),
                ]
            )
            magnitude = np.sqrt(np.sum(difference_vectors**2, axis=0))
            normalized_vectors = difference_vectors / magnitude

            # Find the dot product of the vectors
            qdot = [
                normalized_vectors[0][(i + 1) % 3] * normalized_vectors[0][(i + 2) % 3]
                + normalized_vectors[1][(i + 1) % 3]
                * normalized_vectors[1][(i + 2) % 3]
                + normalized_vectors[2][(i + 1) % 3]
                * normalized_vectors[2][(i + 2) % 3]
                for i in range(3)
            ]

            # Find the index of the vector that is not orthogonal to the others
            non_orthogonal_index = next(
                (
                    index
                    for index, dot_product in enumerate(qdot)
                    if not np.isclose(dot_product, 0)
                ),
                np.argmin(magnitude),
            )

            # Select the two vectors that are not the one found before
            selected_vectors = [
                vec
                for i, vec in enumerate(difference_vectors.T)
                if i != non_orthogonal_index
            ]

            # Calculate the cross product of the selected vectors to find the rotation axis
            rotation_axis = np.cross(selected_vectors[0], selected_vectors[1])

            # Normalize the rotation axis by dividing it by its scale
            normalized_rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Calculate the angles between the normalized rotation axis and the norm1, slip1, and b1 vectors
            angles = np.array(
                [
                    np.arccos(np.dot(vector, normalized_rotation_axis))
                    for vector in [norm1, slip1, b1]
                ]
            )

            # Calculate the absolute differences between the angles and pi/2
            angle_differences = np.abs(angles - np.pi / 2)

            # Select the minimum difference from either the norm or slip axes
            selected_index = np.argmin(angle_differences[:2])

            # Calculate the rotation for the current iteration
            rotation_temp = (
                np.cos(phi[selected_index]) - np.cos(angles[selected_index]) ** 2
            ) / (np.sin(angles[selected_index]) ** 2)

            # Ensure the rotation_temp is within the range [-1, 1]
            rotation_temp = np.clip(rotation_temp, -1, 1)

            # Convert the rotation from radians to degrees
            rotation_degrees = np.rad2deg(np.arccos(rotation_temp))

            # Store the rotation degrees for the current iteration
            rotations[iteration] = rotation_degrees

    # Find the minimum rotation index for the 4 combos to determine the rotation plane
    rotation_index = np.argmin(rotations)
    return int(np.clip(rotation_index, 1, 2))


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


def get_l_w_mag_scaling(
    mag: float, rake: float, tect_class: str
) -> tuple[float, float]:
    """
    Get the length and width of the fault using magnitude scaling

    Parameters
    ----------
    mag : float
        The magnitude of the event
    rake : float
        The rake angle of the fault in degrees
    tect_class : str
        The tectonic class of the event

    Returns
    -------
    length : float
        The length of the fault along strike in km
    dip_dist : float
        The width of the fault down dip in km
    """
    # Load the config
    config = cfg.Config()
    mag_scale_slab_min = config.get_value("mag_scale_slab_min")
    mag_scale_slab_max = config.get_value("mag_scale_slab_max")

    # Get the width and length using magnitude scaling
    if tect_class == "Interface":
        # Use SKARLATOUDIS2016
        dip_dist = np.sqrt(mag_scaling.mw_to_a_skarlatoudis(mag))
        length = dip_dist
    elif tect_class == "Slab" and mag_scale_slab_min <= mag <= mag_scale_slab_max:
        # Use STRASSER2010SLAB
        length = strasser_2010.mw_to_l_strasser_2010_slab(mag)
        dip_dist = strasser_2010.mw_to_w_strasser_2010_slab(mag)
    else:
        # Use LEONARD2014
        length = mag_scaling.mw_to_l_leonard(mag, rake)
        dip_dist = mag_scaling.mw_to_w_leonard(mag, rake)
    return length, dip_dist


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
    ccdl_tect_class = ccldpy.TECTONIC_MAPPING[event_row.tect_class]
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
    strike = selected["Strike"].values[0]
    dip = selected["Dip"].values[0]
    rake = selected["Rake"].values[0]
    length = selected["Length (km)"].values[0]
    dip_dist = selected["Width (km)"].values[0]
    ztor = selected["Rupture Top Depth (km)"].values[0]
    dbottom = selected["Rupture Bottom Depth (km)"].values[0]

    return strike, dip, rake, length, dip_dist, ztor, dbottom


def get_nodal_plane_info(
    event_id: str,
    event_row: pd.Series,
    geonet_cmt_df: pd.DataFrame,
    modified_cmt_df: pd.DataFrame,
    domain_focal_df: pd.DataFrame,
    srf_files: dict,
    rupture_models: dict,
) -> dict:
    """
    Determine the correct nodal plane for the event
    First checks if the event is in the srf_files, if it is, it uses the srf file to determine the nodal plane
    If it is not in the srf_files, it checks if the event is in the rupture_models to determine the nodal plane
    If it is not in the rupture_models, it checks if the event is in the modified CMT data to determine the nodal plane
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
    rupture_models : dict
        The rupture models for specific events

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
    length, dip_dist, srf_points, srf_header, ztor, dbottom = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    # Check if the event_id is in the srf_files
    if event_id in srf_files:
        srf_file = str(srf_files[event_id])
        srf_points = srf.read_srf_points(srf_file)
        srf_header = srf.read_header(srf_file, idx=True)
        f_type = "ff"
        cmt = geonet_cmt_df[geonet_cmt_df.PublicID == event_id].iloc[0]
        ztor = srf_points[0][2]
        dbottom = srf_points[-1][2]

        if len(srf_header) > 1:
            strike, rake, dip, length, dip_dist = None, None, None, None, None
        else:
            fault_strike = srf_header[0]["strike"]
            strike1_diff = abs(fault_strike - cmt.strike1)
            if strike1_diff > 180:
                strike1_diff = 360 - strike1_diff
            strike2_diff = abs(fault_strike - cmt.strike2)
            if strike2_diff > 180:
                strike2_diff = 360 - strike2_diff

            if strike1_diff < strike2_diff:
                strike = cmt.strike1
                rake = cmt.rake1
                dip = cmt.dip1
            else:
                strike = cmt.strike2
                rake = cmt.rake2
                dip = cmt.dip2
            length = srf_header[0]["length"]
            dip_dist = srf_header[0]["width"]
        # length = np.sum([header["length"] for header in srf_header])
        # dip_dist = np.mean([header["width"] for header in srf_header])
    # elif event_id in rupture_models:
    #     # Event is in the rupture models
    #     f_type = "geonet_rm"
    #     rupture_data = geonet_rupture_models.get_seismic_data_from_url(
    #         rupture_models[event_id]
    #     )
    #     (
    #         ztor,
    #         dbottom,
    #         strike,
    #         dip,
    #         rake,
    #         length,
    #         dip_dist,
    #     ) = (
    #         rupture_data["ztor"],
    #         rupture_data["dbottom"],
    #         rupture_data["strike"],
    #         rupture_data["dip"],
    #         rupture_data["rake"],
    #         rupture_data["length"],
    #         rupture_data["width"],
    #     )
    elif event_id in modified_cmt_df.PublicID.values:
        # Event is in the modified CMT data
        f_type = "cmt"
        cmt = modified_cmt_df[modified_cmt_df.PublicID == event_id].iloc[0]
        # Compute the CCLD Simulations for the event
        strike, dip, rake, length, dip_dist, ztor, dbottom = run_ccld_simulation(
            event_id, event_row, cmt.strike1, cmt.dip1, cmt.rake1, "A"
        )
        # strike = cmt.strike1
        # dip = cmt.dip1
        # rake = cmt.rake1

    elif event_id in geonet_cmt_df.PublicID.values:
        # Event is in the Geonet CMT data
        # Need to determine the correct plane
        f_type = "cmt_unc"
        cmt = geonet_cmt_df[geonet_cmt_df.PublicID == event_id].iloc[0]

        # Compute the CCLD Simulations for the event
        strike, dip, rake, length, dip_dist, ztor, dbottom = run_ccld_simulation(
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

        # norm, slip = calc_fnorm_slip(cmt.strike1, cmt.dip1, cmt.rake1)
        #
        # # Get the domain focal values
        # do_strike, do_rake, do_dip = get_domain_focal(
        #     event_row["domain_no"], domain_focal_df
        # )
        #
        # # Figure out the correct plane based on the rotation and the domain focal values
        # do_norm, do_slip = calc_fnorm_slip(do_strike, do_dip, do_rake)
        # plane_out = mech_rot(do_norm, norm, do_slip, slip)
        #
        # if plane_out == 1:
        #     strike, dip, rake = cmt.strike1, cmt.dip1, cmt.rake1
        # else:
        #     strike, dip, rake = cmt.strike2, cmt.dip2, cmt.rake2
    else:
        # Event is not found in any of the datasets
        # Use the domain focal
        f_type = "domain"
        strike, rake, dip = get_domain_focal(event_row["domain_no"], domain_focal_df)

        # Compute the CCLD Simulations for the event
        strike, dip, rake, length, dip_dist, ztor, dbottom = run_ccld_simulation(
            event_id, event_row, strike, dip, rake, "D"
        )

    return {
        "strike": strike,
        "rake": rake,
        "dip": dip,
        "ztor": ztor,
        "dbottom": dbottom,
        "length": length,
        "dip_dist": dip_dist,
        "srf_points": srf_points,
        "srf_header": srf_header,
        "f_type": f_type,
    }


def compute_distances_for_event(
    event_row: pd.Series,
    im_df: pd.DataFrame,
    station_df: pd.DataFrame,
    modified_cmt_df: pd.DataFrame,
    geonet_cmt_df: pd.DataFrame,
    domain_focal_df: pd.DataFrame,
    taupo_polygon: Polygon,
    srf_files: dict,
    rupture_models: dict,
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
    rupture_models : dict
        The rupture models for specific events

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
        rupture_models,
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
    )

    # Get the length and dip_dist if they are None
    # This is the case for when grabbing strike, dip, rake from the CMT files or the domain focal
    # if f_type != "ff" and (length is None or dip_dist is None):
    #     length, dip_dist = get_l_w_mag_scaling(
    #         event_row["mag"], rake, event_row["tect_class"]
    #     )

    # Get the ztor and dbottom if they are None
    # This is the case for when grabbing strike, dip, rake from the CMT files or the domain focal
    # if f_type != "ff" and (ztor is None or dbottom is None):
    #     height = np.sin(np.radians(dip)) * dip_dist
    #     ztor = max(event_row["depth"] - (height / 2), 0)
    #     dbottom = ztor + height

    if srf_header is None or srf_points is None:
        # Calculate the corners of the plane
        dip_dir = (strike + 90) % 360
        projected_width = dip_dist * np.cos(np.radians(dip))
        corner_0, corner_1, corner_2, corner_3 = grid.grid_corners(
            np.asarray([event_row["lat"], event_row["lon"]]),
            strike,
            dip_dir,
            ztor,
            dbottom,
            length,
            projected_width,
        )

        # Utilise grid functions from qcore to get the mesh grid
        config = cfg.Config()
        points_per_km = config.get_value("points_per_km")
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
        srf_header = [{"nstrike": nstrike, "ndip": ndip, "strike": strike}]

        # Divide the srf depth points by 1000
        srf_points[:, 2] /= 1000

    # Calculate the distances
    rrups, rjbs, rrup_points = src_site_dist.calc_rrup_rjb(
        srf_points, stations, return_rrup_points=True
    )
    rxs, rys = src_site_dist.calc_rx_ry(srf_points, srf_header, stations, type=1)
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
            },
        ]
    )

    return propagation_data_combo, extra_event_data, rrup_points


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
    event_df = pd.read_csv(flatfile_dir / "earthquake_source_table.csv")

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

    # Get the srf files as a dictionary of file paths and event_id as the key
    srf_files = {}
    for srf_file in (data_dir / "SrfSourceModels").glob("*.srf"):
        srf_files[srf_file.stem] = srf_file

    # Get the rupture models from Geonet
    # rupture_models = geonet_rupture_models.get_rupture_models()

    # Get the IM data
    im_df = pd.read_csv(flatfile_dir / "ground_motion_im_catalogue.csv")
    # Convert the evid to a string
    im_df["evid"] = im_df["evid"].astype(str)

    # Get the station information
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
    station_df = pd.DataFrame(
        station_info, columns=["net", "sta", "lat", "lon", "elev"]
    )
    station_df = station_df.drop_duplicates().reset_index(drop=True)

    # Select unique stations from IM data and merge
    im_station_df = im_df[["sta"]].drop_duplicates()
    station_df = pd.merge(im_station_df, station_df, on="sta", how="left")
    station_df["depth"] = station_df["elev"] / -1000
    # station_df["depth"] = station_df["elev"] / -1

    # Filter event df to a single event 2016p858000
    # event_df = event_df[event_df.evid == "2016p858000"]

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
                rupture_models=None,
            ),
            [row for idx, row in event_df.iterrows()],
        )

    # Combine the results
    propagation_results, extra_event_results, rrup_points = zip(*result_dfs)
    propagation_data = pd.concat(propagation_results)
    extra_event_data = pd.concat(extra_event_results)

    # Merge the extra event data with the event data
    event_df = pd.merge(event_df, extra_event_data, on="evid", how="right")

    a = np.concatenate(rrup_points)
    df = pd.DataFrame(a)
    df.to_csv(flatfile_dir / "rrup_points.csv", index=False)

    # Save the results
    propagation_data.to_csv(flatfile_dir / "propagation_path_table.csv", index=False)
    event_df.to_csv(flatfile_dir / "earthquake_source_table_adjusted.csv", index=False)
