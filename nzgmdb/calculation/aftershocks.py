from itertools import compress
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import MultiPoint, Point, Polygon

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from qcore import geo
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

    config = cfg.Config()
    ll_num = config.get_value("ll_num")
    nztm_num = config.get_value("nztm_num")
    nztm2wgs = Transformer.from_crs(nztm_num, ll_num)

    # Create all the rupture polygons
    rupture_area_poly = []
    for _, row in catalogue_pd.iterrows():
        if row["evid"] in srf_evids:
            # Generate the convex hull for the SRF
            srf_file = srf_dir / f"{row['evid']}.srf"
            srf_model = srf.read_srf(srf_file)
            # Transform the srf_model geometry to WGS84
            transformed_coords = [
                nztm2wgs.transform(x, y)[::-1]
                for x, y in srf_model.geometry.convex_hull.exterior.coords
            ]
            rupture_area_poly.append(Polygon(transformed_coords))

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

    # Obtain the CRJB cutoffs
    config = cfg.Config()
    crjb_cutoffs = config.get_value("crjb_cutoffs")

    # Iterate over the rjb_cutoff values
    for crjb_cutoff in crjb_cutoffs:
        # Run the abwd_crjb function
        flagvector, vcl = abwd_crjb(
            catalogue_pd,
            rupture_area_poly,
            crjb_cutoff=crjb_cutoff,
        )

        # Store the results in the dictionary
        results_dict[f"aftershock_flag_crjb{crjb_cutoff}"] = flagvector
        results_dict[f"cluster_flag_crjb{crjb_cutoff}"] = vcl

    # Create a DataFrame from the dictionary
    results_df = pd.DataFrame(results_dict)

    merged_df = catalogue_pd.merge(results_df, left_index=True, right_index=True)

    # Save the merged DataFrame
    merged_df.to_csv(
        flatfile_dir
        / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_AFTERSHOCKS,
        index=False,
    )


def decimal_year(catalogue_pd: pd.DataFrame) -> np.ndarray:
    """
    Converts earthquake datetime into decimal years.

    Parameters
    ----------
    catalogue_pd : pandas.DataFrame
        Dataframe containing a 'datetime' column.

    Returns
    -------
    numpy.ndarray
        Array of decimal years.
    """
    time_pd = pd.to_datetime(catalogue_pd["datetime"]).dt.tz_localize(None)

    year = time_pd.dt.year
    start_of_year = pd.to_datetime(year.astype(str) + "-01-01")
    start_of_next_year = pd.to_datetime((year + 1).astype(str) + "-01-01")

    elapsed = (time_pd - start_of_year).dt.total_seconds()
    duration = (start_of_next_year - start_of_year).dt.total_seconds()

    return np.array(year + elapsed / duration)


def resample_polygon_1km(rupture_polygons: list[Polygon]) -> list[MultiPoint]:
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


def calculate_crjb(
    rupture_poly: Polygon, boundary_points: MultiPoint, centroids: np.ndarray
) -> np.ndarray:
    """
    Calculates closest Rupture-to-Just Beyond (CRJB) distance for given earthquake centroids.

    Parameters
    ----------
    rupture_poly : shapely.geometry.Polygon
        Polygon representing the rupture area.
    boundary_points : shapely.geometry.MultiPoint
        Set of resampled boundary points of the rupture.
    centroids : np.ndarray
        Array of centroid coordinates.

    Returns
    -------
    numpy.ndarray
        Array of min CRJB distances.
    """
    # Gather the points of the boundary
    points = np.array([(p.x, p.y) for p in boundary_points.geoms])

    # Calculate the distances from the boundary points to all the centroids
    distances = geo.get_distances(points, centroids[:, 0], centroids[:, 1])

    # Get the minimum distance for each centroid
    if distances.ndim > 1:
        crib_distances_min = np.min(distances, axis=1)
    else:
        crib_distances_min = np.min(distances)

    # Extra check over the rupture polygon to set the CRJB distance to 0
    # if the centroid is inside the rupture
    for i, centroid in enumerate(centroids):
        if rupture_poly.contains(Point(centroid)):
            if isinstance(crib_distances_min, np.ndarray):
                crib_distances_min[i] = 0.0
            else:
                crib_distances_min = 0.0

    return crib_distances_min


def abwd_crjb(
    catalogue_pd: pd.DataFrame,
    rupture_area_poly: list,
    crjb_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies earthquake clusters using spatial and temporal windows.

    Parameters
    ----------
    catalogue_pd : pandas.DataFrame
        Earthquake catalog with at least 'datetime' and 'mag' columns.
    rupture_area_poly : list
        List of rupture polygons (shapely.geometry.Polygon objects).
    crjb_cutoff : float
        Cutoff radius for spatial windowing in km.

    Returns
    -------
    numpy.ndarray
        Vector indicating whether each event is a mainshock (0),
        an aftershock (1)
    numpy.ndarray
        Cluster labels for each earthquake and it's set of aftershocks.
    """
    # Convert datetime to decimal years
    decimal_years = decimal_year(catalogue_pd)

    # Compute rupture centroids and resampled boundaries
    centroids = [[poly.centroid.x, poly.centroid.y] for poly in rupture_area_poly]
    resampled_boundaries = resample_polygon_1km(rupture_area_poly)

    neq = len(catalogue_pd)
    # Note, just a rough approximation is needed for the time window, also used in the GardnerKnopoff method
    DAYS_IN_YEAR = 364.75

    # Define time window based on GardnerKnopoff
    sw_time = np.power(10.0, 0.032 * catalogue_pd.mag + 2.7389) / DAYS_IN_YEAR

    # Adjust the space window for M > 6.5
    sw_time[catalogue_pd.mag < 6.5] = (
        np.power(10.0, 0.5409 * catalogue_pd.mag[catalogue_pd.mag < 6.5] - 0.547)
        / DAYS_IN_YEAR
    )

    eqid = np.arange(neq)
    cluster_labels = np.zeros(neq)
    # Sort indices by magnitude in descending order
    sorted_indices = np.argsort(catalogue_pd.mag, kind="stable")[::-1]

    # Use indexing to sort arrays and lists
    decimal_years = decimal_years[sorted_indices]
    sw_time = sw_time[sorted_indices]
    eqid = eqid[sorted_indices]
    sorted_polygons = np.array(rupture_area_poly)[sorted_indices]
    sorted_centroids = np.array(centroids)[sorted_indices]
    sorted_boundaries = np.array(resampled_boundaries)[sorted_indices]

    flagvector = np.zeros(neq)
    cluster_index = 1

    for i in range(neq - 1):
        if cluster_labels[i] == 0:
            # Find events that fit within the time window
            dt = decimal_years - decimal_years[i]
            valid = (cluster_labels == 0) & (dt >= 0) & (dt <= sw_time[i])

            # calculate the CRJB distances for those events
            aftershock_centroids = np.array(list(compress(sorted_centroids, valid)))
            crjb_distances = calculate_crjb(
                sorted_polygons[i], sorted_boundaries[i], aftershock_centroids
            )

            # Only allow valid aftershocks for those within the CRJB cutoff
            valid[valid] = crjb_distances <= crjb_cutoff

            # Set the mainshock and aftershock flags as well as the cluster labels
            valid[i] = False
            if valid.any():
                cluster_labels[valid] = cluster_index
                cluster_labels[i] = cluster_index
                flagvector[valid] = 1
                cluster_index += 1

    # Sort the results back to the original order
    return flagvector[np.argsort(eqid)], cluster_labels[np.argsort(eqid)]
