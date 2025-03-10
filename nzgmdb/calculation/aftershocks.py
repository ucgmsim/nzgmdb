from itertools import compress
from pathlib import Path

import alphashape
import numpy as np
import pandas as pd
from shapely import MultiPoint, Polygon

from abwd_declust.abwd_declust_v2_1 import abwd_crjb
from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from source_modelling import srf


def merge_aftershocks(main_dir: Path):
    """
    Merge the aftershock flags and cluster flags into the earthquake source table

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    """
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    catalogue_pd = pd.read_csv(
        flatfile_dir
        / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_DISTANCES,
        dtype={"evid": str},
    )

    # Apply % 360 to manage the longitude negative values
    catalogue_pd.loc[
        :, ["corner_0_lon", "corner_1_lon", "corner_2_lon", "corner_3_lon", "hyp_lon"]
    ] %= 360

    # Get the SRF source models
    data_dir = file_structure.get_data_dir()
    srf_dir = data_dir / "SrfSourceModels"
    srf_evids = [srf_file.stem for srf_file in srf_dir.glob("*.srf")]

    # Create all the rupture polygons
    rupture_area_poly = []
    for _, row in catalogue_pd.iterrows():
        if row["evid"] in srf_evids:
            # Generate the convex hull for the SRF
            srf_file = srf_dir / f"{row['evid']}.srf"
            srf_model = srf.read_srf(srf_file)
            srf_points = srf_model.points.loc[:, ["lon", "lat", "dep"]].to_numpy()
            # Apply % 360 to manage the longitude negative values
            srf_points[:, 0] = srf_points[:, 0] % 360
            nodal_plane_vertices = np.transpose(
                np.array([srf_points[:, 0], srf_points[:, 1]])
            )
            alphashape2 = alphashape.alphashape(nodal_plane_vertices, 0.0)
            hull_x, hull_y = alphashape2.exterior.xy
            rupture_area_poly.append(Polygon(zip(hull_x, hull_y)))
        else:
            lon_values = [
                row["corner_0_lon"],
                row["corner_1_lon"],
                row["corner_3_lon"],
                row["corner_2_lon"],
                row["corner_0_lon"],
            ]
            lat_values = [
                row["corner_0_lat"],
                row["corner_1_lat"],
                row["corner_3_lat"],
                row["corner_2_lat"],
                row["corner_0_lat"],
            ]

            # Create a Polygon from the corner coordinates
            rupture_area_temp = Polygon(zip(lon_values, lat_values))
            rupture_area_poly.append(rupture_area_temp)

    # Initialize an empty dictionary to store the results
    results_dict = {}

    # Obtain the RJB cutoffs
    config = cfg.Config()
    rjb_cutoffs = config.get_value("rjb_cutoffs")

    # Iterate over the rjb_cutoff values
    for rjb_cutoff in rjb_cutoffs:
        # Run the abwd_crjb function
        flagvector, vcl = abwd_crjb(
            catalogue_pd,
            rupture_area_poly,
            rjb_cutoff=rjb_cutoff,
            window_method="GardnerKnopoff",
            fs_time_prop=0,
        )

        # Store the results in the dictionary
        results_dict[f"aftershock_flag_crjb{rjb_cutoff}"] = flagvector
        results_dict[f"cluster_flag_crjb{rjb_cutoff}"] = vcl

    # Create a DataFrame from the dictionary
    results_df = pd.DataFrame(results_dict)

    merged_df = catalogue_pd.merge(results_df, left_index=True, right_index=True)

    # Save the merged DataFrame
    merged_df.to_csv(
        flatfile_dir
        / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_AFTERSHOCKS,
        index=False,
    )


import time

from shapely.geometry import MultiPoint, Point


def abwd_crjb(catalogue_pd, rupture_area_poly, rjb_cutoff, window_method, fs_time_prop):
    """
    Identifies earthquake clusters using spatial and temporal windows.

    Parameters
    ----------
    catalogue_pd : pandas.DataFrame
        Earthquake catalog with at least 'datetime' and 'mag' columns.
    rupture_area_poly : list
        List of rupture polygons (shapely.geometry.Polygon objects).
    rjb_cutoff : float
        Cutoff radius for spatial windowing in km.
    window_method : str
        Method to define the temporal and spatial clustering windows.
        Options: "GardnerKnopoff", "Gruenthal", "Urhammer".
    fs_time_prop : float
        Fore-shock time proportion for temporal window.

    Returns
    -------
    flagvector : numpy.ndarray
        Vector indicating whether each event is a mainshock (0),
        an aftershock (1), or a foreshock (-1).
    vcl : numpy.ndarray
        Cluster labels for each earthquake (0 means no cluster).
    """

    def decimal_year(catalogue_pd):
        """
        Converts earthquake datetime into decimal years.

        Parameters
        ----------
        catalogue_pd : pandas.DataFrame
            Dataframe containing a 'datetime' column.

        Returns
        -------
        numpy.ndarray
            Array of decimal year representations.
        """

        def to_year_fraction(date):
            def since_epoch(date):
                return time.mktime(date.timetuple())

            year = date.year
            start_of_year = dt(year=year, month=1, day=1)
            start_of_next_year = dt(year=year + 1, month=1, day=1)

            elapsed = since_epoch(date) - since_epoch(start_of_year)
            duration = since_epoch(start_of_next_year) - since_epoch(start_of_year)

            return year + elapsed / duration

        time_pd = pd.to_datetime(catalogue_pd["datetime"], format="ISO8601")
        return np.array([to_year_fraction(date) for date in time_pd])

    def haversine_distance(
        lon1, lat1, lon2, lat2, radians=False, earth_radius=6371.227
    ):
        """
        Calculates great-circle distance using the haversine formula.

        Parameters
        ----------
        lon1, lat1 : numpy.ndarray
            Arrays of longitudes and latitudes for the first set of points.
        lon2, lat2 : float
            Longitude and latitude of the second point.
        radians : bool, optional
            If True, inputs are in radians. Default is False.
        earth_radius : float, optional
            Radius of the Earth in km. Default is 6371.227 km.

        Returns
        -------
        numpy.ndarray
            Array of distances in km.
        """
        if not radians:
            conversion_factor = np.pi / 180.0
            lon1, lat1, lon2, lat2 = conversion_factor * np.array(
                [lon1, lat1, lon2, lat2]
            )

        dlat = lat1 - lat2
        dlon = lon1 - lon2
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        return 2.0 * earth_radius * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def rupture_centroids(rupture_polygons):
        """
        Computes centroid coordinates for each rupture polygon.

        Parameters
        ----------
        rupture_polygons : list
            List of shapely.geometry.Polygon objects.

        Returns
        -------
        list of [float, float]
            List of centroid coordinates (longitude, latitude).
        """
        return [[poly.centroid.x, poly.centroid.y] for poly in rupture_polygons]

    def resample_polygon_1km(rupture_polygons):
        """
        Resamples polygon boundaries at approximately 1 km resolution.

        Parameters
        ----------
        rupture_polygons : list
            List of shapely.geometry.Polygon objects.

        Returns
        -------
        list
            List of shapely.geometry.MultiPoint objects containing resampled boundary points.
        """
        resampled_points = []

        for poly in rupture_polygons:
            x, y = poly.exterior.xy
            xy = np.column_stack([x, y])
            num_points = int(poly.length * 111.2) + 8

            distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)])
            sampled_distances = np.linspace(0, distances.max(), num_points)
            interpolated_xy = np.column_stack(
                [
                    np.interp(sampled_distances, distances, xy[:, 0]),
                    np.interp(sampled_distances, distances, xy[:, 1]),
                ]
            )

            resampled_points.append(MultiPoint(interpolated_xy))

        return resampled_points

    def calculate_crjb(rupture_poly, boundary_points, centroids):
        """
        Calculates closest Rupture-to-Just Beyond (CRJB) distance for given earthquake centroids.

        Parameters
        ----------
        rupture_poly : shapely.geometry.Polygon
            Polygon representing the rupture area.
        boundary_points : shapely.geometry.MultiPoint
            Set of resampled boundary points of the rupture.
        centroids : list
            List of centroid coordinates.

        Returns
        -------
        numpy.ndarray
            Array of CRJB distances.
        """
        crjb_distances = []

        for centroid in centroids:
            if rupture_poly.contains(Point(centroid)):
                crjb_distances.append(0.0)
            else:
                distances = [
                    haversine_distance(centroid[0], centroid[1], p.x, p.y)
                    for p in boundary_points.geoms
                ]
                crjb_distances.append(min(distances))

        return np.array(crjb_distances)

    # Convert datetime to decimal years
    decimal_years = decimal_year(catalogue_pd)

    # Compute rupture centroids and resampled boundaries
    centroids = rupture_centroids(rupture_area_poly)
    resampled_boundaries = resample_polygon_1km(rupture_area_poly)

    neq = len(catalogue_pd)
    DAYS_IN_YEAR = 364.75

    # Define time and space windows based on method
    if window_method == "GardnerKnopoff":
        sw_time = (
            np.exp(-3.95 + np.sqrt(0.62 + 17.32 * catalogue_pd.mag)) / DAYS_IN_YEAR
        )
    elif window_method == "Gruenthal":
        sw_time = (
            np.exp(-3.95 + np.sqrt(0.62 + 17.32 * catalogue_pd.mag)) / DAYS_IN_YEAR
        )
    elif window_method == "Urhammer":
        sw_time = np.exp(-2.87 + 1.235 * catalogue_pd.mag) / DAYS_IN_YEAR
    else:
        sw_time = (
            np.exp(-3.95 + np.sqrt(0.62 + 17.32 * catalogue_pd.mag)) / DAYS_IN_YEAR
        )

    sw_space = np.full(neq, rjb_cutoff)

    eqid = np.arange(neq)
    cluster_labels = np.zeros(neq, dtype=int)
    sorted_indices = np.argsort(catalogue_pd.mag)[::-1]

    decimal_years = decimal_years[sorted_indices]
    sw_time = sw_time[sorted_indices]
    eqid = eqid[sorted_indices]

    sorted_polygons = [rupture_area_poly[i] for i in sorted_indices]
    sorted_centroids = [centroids[i] for i in sorted_indices]
    sorted_boundaries = [resampled_boundaries[i] for i in sorted_indices]

    flagvector = np.zeros(neq, dtype=int)
    cluster_index = 0

    for i in range(neq - 1):
        if cluster_labels[i] == 0:
            dt = decimal_years - decimal_years[i]
            valid = (
                (cluster_labels == 0)
                & (-sw_time[i] * fs_time_prop <= dt)
                & (dt <= sw_time[i])
            )

            aftershock_centroids = list(compress(sorted_centroids, valid))
            crjb_distances = calculate_crjb(
                sorted_polygons[i], sorted_boundaries[i], aftershock_centroids
            )
            valid[valid] = crjb_distances <= sw_space[i]

            if valid.any():
                cluster_labels[valid] = cluster_index + 1
                flagvector[valid] = 1
                cluster_index += 1

    return flagvector[np.argsort(eqid)], cluster_labels[np.argsort(eqid)]
