from pathlib import Path
from math import sin, cos, acos, radians

import numpy as np
import pandas as pd
from pyproj import Transformer
from obspy.clients.fdsn import Client as FDSN_Client
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon, LineString

from qcore import srf, nhm, geo, grid
from qcore.uncertainties import mag_scaling
from qcore.uncertainties.magnitude_scaling import strasser_2010
from IM_calculation.source_site_dist import src_site_dist
from nzgmdb.data_retrieval import focal
from nzgmdb.management import file_structure, config as cfg


def merge_im_data(
    main_dir: Path,
    gmc_ffp: Path,
    fmax_ffp: Path,
):
    """
    Merge the IM data into a single flatfile
    """
    # Get the flatfile directory
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    # Get the IM directory
    im_dir = file_structure.get_im_dir(main_dir)

    # Load the GMC file
    gmc_results = pd.read_csv(gmc_ffp)

    # Load the fmax file
    fmax_results = pd.read_csv(fmax_ffp)

    # Define the columns to be grouped
    columns = ["score_mean", "fmin_mean", "multi_mean"]

    # Group by 'record' and 'component', then aggregate the columns
    new_df = gmc_results.groupby(["record", "component"])[columns].mean().unstack()

    # Join the column names to score_mean_X etc.
    new_df.columns = ["_".join(col) for col in new_df.columns]

    new_df = new_df.reset_index()

    # Find all the IM files
    im_files = im_dir.rglob("*IM.csv")

    # Concat all the IM files
    im_all = pd.concat([pd.read_csv(file) for file in im_files])

    # Merge the gm_all and new_df on record
    gm_final = pd.merge(
        im_all,
        new_df,
        left_on="record_id",
        right_on="record",
        how="left",
    )

    # Add the chan, loc and rename event_id and station across the entire series
    gm_final[["evid", "sta", "chan", "loc"]] = gm_final["record_id"].str.split(
        "_", expand=True
    )

    # remove the record column
    gm_final = gm_final.drop(columns=["record"])

    # Get the Ds595 Lower Bound
    config = cfg.Config()
    Ds595_lower_bound = config.get_value("Ds595_lower_bound")

    # Filter the Ds595 by first filtering only the ver, 000 and 090 components
    comp_sub = gm_final[gm_final.component.isin(["ver", "000", "090"])]
    # Then Sum the Ds595 values for each record
    comp_sub_grouped = comp_sub.groupby(["record_id"]).sum()
    # Find the records that are below the Ds595 lower bound
    Ds595_filter_records = comp_sub_grouped[
        comp_sub_grouped.Ds595 < Ds595_lower_bound
    ].reset_index()[["record_id"]]
    # Remove the records that are below the Ds595 lower bound
    gm_final = gm_final[~gm_final.record_id.isin(Ds595_filter_records.record_id)]

    # Merge in fmax TODO
    gm_final = pd.merge(
        gm_final,
        fmax_results,
        left_on="record_id",
        right_on="record",
        how="left",
    )

    # Sort columns nicely
    gm_final = gm_final[
        [
            "record_id",
            "evid",
            "sta",
            "loc",
            "chan",
            "component",
            "PGA",
            "PGV",
            "CAV",
            "AI",
            "Ds575",
            "Ds595",
            "MMI",
            "score_mean_X",
            "score_mean_Y",
            "score_mean_Z",
            "multi_mean_X",
            "multi_mean_Y",
            "multi_mean_Z",
            "fmin_mean_X",
            "fmin_mean_Y",
            "fmin_mean_Z",
            "pSA",
            "FAS",
        ]
    ]

    # Save the ground_motion_im_catalogue.csv
    gm_final.to_csv(flatfile_dir / "ground_motion_im_catalogue.csv", index=False)


def calc_fnorm_slip(strike: float, dip: float, rake: float):
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
    """
    phi = np.deg2rad(strike)
    delt = np.deg2rad(dip)
    lam = np.deg2rad(rake)

    fnorm = -sin(delt) * sin(phi), sin(delt) * cos(phi), -cos(delt)
    slip = (
        cos(lam) * cos(phi) + cos(delt) * sin(lam) * sin(phi),
        cos(lam) * sin(phi) - cos(delt) * sin(lam) * cos(phi),
        -sin(lam) * sin(delt),
    )

    return fnorm, slip


def mech_rot(norm1, norm2, slip1, slip2):
    B1 = np.cross(norm1, slip1)

    rotations = np.zeros(4)
    for iteration in range(0, 4):
        if iteration < 2:
            norm2_temp = norm2
            slip2_temp = slip2
        else:
            norm2_temp = slip2
            slip2_temp = norm2
        if (iteration == 1) or (iteration == 3):
            norm2_temp = tuple(-x for x in norm2_temp)
            slip2_temp = tuple(-x for x in slip2_temp)

        B2 = np.cross(norm2_temp, slip2_temp)

        phi1 = np.dot(norm1, norm2_temp)
        phi2 = np.dot(slip1, slip2_temp)
        phi3 = np.dot(B1, B2)

        # In some cases, identical dot products produce values incrementally higher than 1
        phi1 = acos(max(min(phi1, 1), -1))
        phi2 = acos(max(min(phi2, 1), -1))
        phi3 = acos(max(min(phi3, 1), -1))

        # Set array of phi for indexing reasons later
        phi = [phi1, phi2, phi3]

        # If the mechanisms are very close, rotation = 0
        epsilon = 1e-4
        if phi1 < epsilon and phi2 < epsilon and phi3 < epsilon:
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
                    np.array(B1) - np.array(B2),
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
                    if dot_product > 0.9999
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

            # Calculate the angles between the normalized rotation axis and the norm1, slip1, and B1 vectors
            angles = np.array(
                [
                    np.arccos(np.dot(vector, normalized_rotation_axis))
                    for vector in [norm1, slip1, B1]
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
    if rotation_index < 2:
        plane_out = 1
    else:
        plane_out = 2

    return plane_out


def get_domain_focal(domain_no: int, domain_focal_df: pd.DataFrame):
    if domain_no == 0:
        return 220, 45, 90
    else:
        domain = domain_focal_df[domain_focal_df.Domain_No == domain_no].iloc[0]
        return domain.strike, domain.rake, domain.dip


def get_l_w_mag_scaling(mag: float, rake: float, tect_class: str):
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


def compute_row(
    event_row: pd.Series,
    im_df: pd.DataFrame,
    station_df: pd.DataFrame,
    modified_cmt_df: pd.DataFrame,
    geonet_cmt_df: pd.DataFrame,
    domain_focal_df: pd.DataFrame,
    taupo_polygon: Polygon,
    srf_files: dict,
    rupture_models: dict,
):
    """
    Compute the distances
    """
    # Extract out the relevant event_row data
    event_id = event_row["evid"]
    im_event_df = im_df[im_df["evid"] == event_id]

    # Check if the event doesn't have IM data
    # If it doesn't, skip the event
    if im_event_df.empty:
        return

    # Get the station data
    event_sta_df = station_df[station_df["sta"].isin(im_event_df["sta"])].reset_index()
    stations = event_sta_df[["lon", "lat", "depth"]].to_numpy()

    length, dip_dist, srf_points, srf_header = None, None, None, None

    # Check if the event_id is in the srf_files
    if event_id in srf_files:
        srf_file = str(srf_files[event_id])
        srf_points = srf.read_srf_points(srf_file)
        srf_header = srf.read_header(srf_file, idx=True)
        f_type = "ff"
        cmt = geonet_cmt_df[geonet_cmt_df.PublicID == event_id].iloc[0]

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
        length = np.sum([header["length"] for header in srf_header])
        dip_dist = np.mean([header["width"] for header in srf_header])
    elif event_id in rupture_models:
        # Event is in the rupture models
        f_type = "geonet_rm"
        (
            ztor,
            dbottom,
            strike,
            dip,
            rake,
            length,
            dip_dist,
        ) = focal.get_seismic_data_from_url(rupture_models[event_id])
    elif event_id in modified_cmt_df.PublicID.values:
        # Event is in the modified CMT data
        f_type = "cmt"
        cmt = modified_cmt_df[modified_cmt_df.PublicID == event_id].iloc[0]
        strike = cmt.strike1
        dip = cmt.dip1
        rake = cmt.rake1
    elif event_id in geonet_cmt_df.PublicID.values:
        # Event is in the Geonet CMT data
        # Need to determine the correct plane
        f_type = "cmt_unc"
        cmt = geonet_cmt_df[geonet_cmt_df.PublicID == event_id].iloc[0]
        norm, slip = calc_fnorm_slip(cmt.strike1, cmt.dip1, cmt.rake1)

        # Get the domain focal values
        do_strike, do_rake, do_dip = get_domain_focal(
            event_row["domain_no"], domain_focal_df
        )

        # Figure out the correct plane based on the rotation and the domain focal values
        do_norm, do_slip = calc_fnorm_slip(do_strike, do_dip, do_rake)
        plane_out = mech_rot(do_norm, norm, do_slip, slip)

        if plane_out == 1:
            strike, dip, rake = cmt.strike1, cmt.dip1, cmt.rake1
        else:
            strike, dip, rake = cmt.strike2, cmt.dip2, cmt.rake2
    else:
        # Event is not found in any of the datasets
        # Use the domain focal
        f_type = "domain"
        strike, rake, dip = get_domain_focal(event_row["domain_no"], domain_focal_df)

    # Get the length and dip_dist if they are None
    # This is the case for when grabbing strike, dip, rake from the CMT files or the domain focal
    if length is None or dip_dist is None:
        length, dip_dist = get_l_w_mag_scaling(
            event_row["mag"], rake, event_row["tect_class"]
        )

    if srf_header is None or srf_points is None:
        # Calculate the corners of the plane
        dip_dir = (strike + 90) % 360
        height = sin(radians(dip)) * dip_dist
        dtop = max(0, event_row["depth"] - (height / 2))
        dbottom = dtop + height
        corner_0, corner_1, corner_2, corner_3 = grid.grid_corners(
            np.asarray([event_row["lat"], event_row["lon"]]),
            strike,
            dip_dir,
            dtop,
            dbottom,
            length,
            dip_dist,
        )

        # Utilise grid functions from qcore to get the mesh grid
        config = cfg.Config()
        points_per_km = config.get_value("points_per_km")
        srf_points = grid.coordinate_meshgrid(
            corner_0, corner_1, corner_2, 1000 / points_per_km
        )
        # Reshape to (n, 3)
        srf_points = srf_points.reshape(-1, 3)

        # Generate the srf header
        nstrike = int(round(length * points_per_km))
        ndip = int(round(dip_dist * points_per_km))
        srf_header = [{"nstrike": nstrike, "ndip": ndip, "strike": strike}]

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
    tvz_lengths, boundary_dists_rjb = TVZ_path_calc(
        event_sta_df,
        taupo_polygon,
        rjbs,
        rrups_lon,
        rrups_lat,
    )
    print(tvz_lengths)


def TVZ_path_calc(
    sta_df,
    taupo_polygon,
    r_epis,
    rrups_lon,
    rrups_lat,
):
    """
    Figures out if the path from the station to the event goes through the Taupo VZ
    And if it does to calculate the length of the path that goes through the Taupo VZ
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
                            np.sqrt((point.x - sta_transform[0]) ** 2 + (point.y - sta_transform[1]) ** 2)
                            / 1000
                    )

            line_points = line.intersection(taupo_polygon)
            tvz_length = min(line_points.length / 1000 / r_epis[station_index], 1)

        tvz_lengths.append(tvz_length)
        boundary_dists_rjb.append(boundary_dist_rjb)

    return tvz_lengths, boundary_dists_rjb


def calc_distances(main_dir: Path):
    """
    Stuff
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
    event_df = pd.read_csv(flatfile_dir / "earthquake_source_table_tectdomain.csv")

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
    rupture_models = focal.get_rupture_models()

    # Get the IM data
    im_df = pd.read_csv(flatfile_dir / "ground_motion_im_catalogue.csv")

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

    # Iterate over the event data
    for idx, event_row in event_df.iterrows():
        compute_row(
            event_row,
            im_df,
            station_df,
            modified_cmt_df,
            geonet_cmt_df,
            domain_focal_df,
            taupo_polygon,
            srf_files,
            rupture_models,
        )


# merge_im_data(
#     main_dir=Path("/home/joel/local/gmdb/US_stuff/new_struct_2022"),
#     gmc_ffp=Path(
#         "/home/joel/local/gmdb/US_stuff/new_struct_2022/flatfiles/gmc_predictions.csv"
#     ),
#     fmax_ffp=Path("/home/joel/local/gmdb/US_stuff/new_struct_2022/flatfiles/fmax.csv"),
# )

# calc_distances(main_dir=Path("/home/joel/local/gmdb/US_stuff/new_struct_2022"))
calc_distances(main_dir=Path("X:/Work/nzgmdb/2022_full_test"))
