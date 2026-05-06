"""
Creates the site table for the NZGMDB. This module fetches the station information from the FDSN clients, and the
Geonet metadata summary information.
"""

from pathlib import Path

import fiona
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

from nzgmdb.data_retrieval import tect_domain, inventory_xml
from nzgmdb.management import config as cfg
from nzgmdb.management.data_registry import NZGMDB_DATA
from qcore import point_in_polygon
from velocity_modelling import registry, threshold


def fill_gaps_with_nearest(
    coords: np.ndarray,
    values: np.ndarray,
    invalid_mask: np.ndarray | None = None,
    k: int = 8,
) -> np.ndarray:
    """
    Fill NaN or invalid values using nearest-neighbour averaging.

    Parameters
    ----------
    coords : (N, 2) array_like
        Coordinates of the points (e.g., [x, y] or [lon, lat]).
    values : (N,) array_like
        Values at the points, with NaN for invalid/missing values.
    invalid_mask : (N,) array_like, optional
        Boolean mask indicating invalid points. If None, NaNs in `values` are used.
    k : int, default=8
        Number of nearest neighbors to consider for averaging.

    Returns
    -------
    ndarray
        Values with NaNs filled using nearest-neighbour averaging.
    """

    coords = np.asarray(coords)
    values = np.asarray(values).astype(float)

    # ---- Enforce correct shapes ----
    if values.ndim == 2 and values.shape[1] == 1:
        values = values.ravel()

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")

    if invalid_mask is None:
        invalid_mask = np.isnan(values)
    else:
        invalid_mask = np.asarray(invalid_mask)
        if invalid_mask.ndim == 2 and invalid_mask.shape[1] == 1:
            invalid_mask = invalid_mask.ravel()

    valid_mask = ~invalid_mask

    if not valid_mask.any():
        return np.full_like(values, np.nan)

    # ---- Build KDTree ----
    tree = cKDTree(coords[valid_mask])
    valid_values = values[valid_mask]

    # ---- Fill invalid points ----
    for idx in np.where(invalid_mask)[0]:
        coord = coords[idx]
        kk = min(k, len(valid_values))
        _, nn = tree.query(coord, k=kk)
        values[idx] = np.nanmean(valid_values[nn])

    return values


def sample_points_from_geotiff(
    file_path: Path,
    latlon_points: np.ndarray,
    band: int = 1,
) -> np.ndarray:
    """
    Sample a GeoTIFF raster at given latitude/longitude points.

    Parameters
    ----------
    file_path : Path
        Path to the GeoTIFF file.
    latlon_points : (N, 2) array_like
        Input points as [lat, lon] in EPSG:4326.
    band : int, default=1
        Raster band to sample (1-based index).

    Returns
    -------
    ndarray
        Sampled raster values. NaN where points fall outside the raster
        or where raster contains nodata.
    """

    # ---- Normalize inputs ----
    file_path = Path(file_path)
    latlon_points = np.asarray(latlon_points, dtype=float)

    lat = latlon_points[:, 0]
    lon = latlon_points[:, 1]

    # Prepare output (NaN by default)
    samples = np.full(lat.shape, np.nan, dtype=float)

    # ---- Open raster ----
    with rasterio.open(file_path) as ds:

        if ds.crs is None:
            raise ValueError("Raster CRS is undefined.")

        # CRS of input coordinates (WGS84 lat/lon)
        input_crs = rasterio.crs.CRS.from_epsg(4326)

        # ---- Transform coordinates if needed ----
        if ds.crs == input_crs:
            # Raster already in lat/lon
            x = lon
            y = lat
        else:
            # Transform lat/lon → raster CRS
            transformer = Transformer.from_crs(
                input_crs,
                ds.crs,
                always_xy=True,
            )
            x, y = transformer.transform(lon, lat)

        # ---- Determine which points lie inside raster bounds ----
        bounds = ds.bounds
        inside = (
            (x >= bounds.left)
            & (x <= bounds.right)
            & (y >= bounds.bottom)
            & (y <= bounds.top)
        )

        if not np.any(inside):
            return samples.reshape(-1, 1)

        # ---- Sample raster at valid points ----
        coords = list(zip(x[inside], y[inside]))

        raw_values = np.array(
            [v[0] for v in ds.sample(coords, indexes=band)],
            dtype=float,
        )

        # ---- Handle nodata ----
        if ds.nodata is not None:
            raw_values[raw_values == ds.nodata] = np.nan

        # ---- Apply scale and offset if defined ----
        scale = 1.0
        offset = 0.0

        if ds.scales is not None:
            scale = ds.scales[band - 1]

        if ds.offsets is not None:
            offset = ds.offsets[band - 1]

        values = raw_values * scale + offset

        # ---- Insert values back into output ----
        samples[inside] = values

    return samples.reshape(-1, 1)


def create_site_table_response(
    add_tmp_arrays: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the site table for the NZGMDB. This function fetches the station information from the FDSN clients, and the
    Geonet metadata summary information. It then merges the two dataframes and determines the tectonic domain for each
    station. The final dataframe is saved as a csv file in the flatfile directory.

    Parameters
    ----------
    add_tmp_arrays : bool, optional
        Whether to add temporary arrays to the station information, by default False

    Returns
    -------
    pd.DataFrame
        The site table dataframe with all Z, vs30, domain and location values for each site
        used in the NZGMDB
    pd.DataFrame
        The station table dataframe with all channel and location values for each site
    """
    # Fetch the client station information
    config = cfg.Config()
    all_info_df = inventory_xml.get_full_inventory(
        add_tmp_arrays=add_tmp_arrays, return_df=True
    )

    # Get the Geonet metadata summary information
    geo_meta_summary_df = pd.read_csv(
        NZGMDB_DATA.fetch("Geonet_Metadata_Summary_v1.4.csv")
    )

    # Rename the columns
    geo_meta_summary_df = geo_meta_summary_df.rename(
        columns={
            "Name": "sta",
            "NZS1170SiteClass": "site_class",
            "Vs30_median": "Vs30",
            "Sigmaln_Vs30": "Vs30_std",
            "T_median": "T0",
            "sigmaln_T": "T0_std",
            "Q_T": "Q_T0",
            "D_T": "D_T0",
            "T_Ref": "T0_ref",
            "Z1.0_median": "Z1.0",
            "sigmaln_Z1.0": "Z1.0_std",
            "Z1.0_Ref": "Z1.0_ref",
            "Z2.5_median": "Z2.5",
            "sigmaln_Z2.5": "Z2.5_std",
            "Z2.5_Ref": "Z2.5_ref",
        }
    )

    for col in ("start_time", "end_time"):
        all_info_df[col] = pd.to_datetime(all_info_df[col], format="ISO8601")

    # Remove the duplicated stations between different networks
    all_info_df = all_info_df.drop_duplicates(
        subset=[
            "sta",
            "lat",
            "lon",
            "elev",
            "chan",
            "loc",
            "loc_elev",
            "start_time",
            "end_time",
        ]
    )

    # separate into site and sta here to avoid merging issues exploding
    site_df = all_info_df[
        ["provider", "net", "sta", "lat", "lon", "elev", "creation_date", "end_date"]
    ]

    merged_df = site_df.merge(
        geo_meta_summary_df,
        on="sta",
        how="left",
    )

    # Specify the required files for fiona
    NZGMDB_DATA.fetch("nt_domains_kiran.shp")
    NZGMDB_DATA.fetch("nt_domains_kiran.dbf")
    NZGMDB_DATA.fetch("nt_domains_kiran.shx")

    # Shape file for determining neotectonic domain
    with fiona.open(Path(NZGMDB_DATA.abspath) / "nt_domains_kiran.shp") as collection:
        shapes = list(collection)
    tect_merged_df = tect_domain.find_domain_from_shapes(merged_df, shapes)

    # Rename the domain column
    tect_merged_df = tect_merged_df.rename(columns={"domain_no": "site_domain_no"})

    # Only compute thresholds for stations where Z1.0 is missing
    mask_missing_z1 = tect_merged_df["Z1.0"].isna()
    mask_q3 = tect_merged_df["Q_Z1.0"] == "Q3"
    mask_to_compute = mask_missing_z1 | mask_q3
    if mask_to_compute.any():
        # Prepare stations DataFrame for only missing rows, indexed by station code
        stations = tect_merged_df.loc[mask_to_compute, ["sta", "lon", "lat"]].set_index(
            "sta"
        )[["lon", "lat"]]
        try:
            nzcvm_version = config.get_value("nzcvm_version")
            thresholds = threshold.compute_station_thresholds(
                stations, model_version=nzcvm_version
            )
            # Merge computed thresholds back (computed columns will be suffixed)
            tect_merged_df = tect_merged_df.merge(
                thresholds[["Z1.0(km)", "Z2.5(km)", "sigma"]],
                left_on="sta",
                right_index=True,
                how="left",
            )

            # Add in the computed values where missing
            tect_merged_df["Z1.0"] = tect_merged_df["Z1.0"].combine_first(
                tect_merged_df.get("Z1.0(km)") * 1000.0
            )
            tect_merged_df["Z2.5"] = tect_merged_df["Z2.5"].combine_first(
                tect_merged_df.get("Z2.5(km)") * 1000.0
            )
            tect_merged_df["Z1.0_std"] = tect_merged_df["Z1.0_std"].combine_first(
                tect_merged_df.get("sigma")
            )
            tect_merged_df["Z2.5_std"] = tect_merged_df["Z2.5_std"].combine_first(
                tect_merged_df.get("sigma")
            )

            # Set extra ref / quality fields
            tect_merged_df.loc[
                mask_to_compute, ["Z1.0_ref", "Z2.5_ref", "Q_Z1.0", "Q_Z2.5"]
            ] = ["NZCVM (2026)", "NZCVM (2026)", "Q3", "Q3"]

            # Get the file path to the combined MVN GeoTIFF
            NZGMDB_DATA.fetch("nzcvm_v1.tif")
            file_path = Path(NZGMDB_DATA.abspath) / "nzcvm_v1.tif"

            # Compute Vs30 for missing values
            points = tect_merged_df.loc[mask_to_compute, ["lat", "lon"]].to_numpy()
            vs30_values = sample_points_from_geotiff(file_path, points).ravel()

            # Fill missing gaps in Vs30 using nearest-neighbour averaging
            coords = np.column_stack([points[:, 1], points[:, 0]])
            vs30_values_filled = fill_gaps_with_nearest(coords, vs30_values)
            vs30_values_filled_rounded = np.round(vs30_values_filled)

            # Update Vs30 and related fields
            tect_merged_df.loc[mask_to_compute, "Vs30"] = vs30_values_filled_rounded

            # Ensure reference and quality fields are set for Vs30 where filled
            vs30_mask = mask_to_compute & ~tect_merged_df["Vs30"].isna()
            tect_merged_df.loc[vs30_mask, "Vs30_Ref"] = "Vs30 Map v1.0 (2026)"
            tect_merged_df.loc[vs30_mask, "Q_Vs30"] = "Q3"

        except (FileNotFoundError, ValueError, RuntimeError):
            raise UserWarning(
                "Could not compute thresholds for missing Z1.0 values, check correct setup for NZCVM"
            )

    # Split into station and site dfs
    station_df = all_info_df.loc[
        :,
        [
            "provider",
            "net",
            "sta",
            "lat",
            "lon",
            "elev",
            "chan",
            "loc",
            "loc_elev",
            "start_time",
            "end_time",
        ],
    ]
    # Adjust any "" loc codes to be "00"
    # based on the FDSN Source Indentifiers documentation (https://docs.fdsn.org/projects/source-identifiers/en/latest/location-codes.html)
    station_df = station_df.replace({"loc": {"": "00"}})

    site_df = tect_merged_df.loc[
        :,
        [
            "provider",
            "net",
            "sta",
            "lat",
            "lon",
            "elev",
            "creation_date",
            "end_date",
            "site_class",
            "Vs30",
            "Vs30_std",
            "Q_Vs30",
            "Vs30_Ref",
            "T0",
            "T0_std",
            "Q_T0",
            "D_T0",
            "T0_ref",
            "Z1.0",
            "Z1.0_std",
            "Q_Z1.0",
            "Z1.0_ref",
            "Z2.5",
            "Z2.5_std",
            "Q_Z2.5",
            "Z2.5_ref",
            "site_domain_no",
        ],
    ]
    site_df = site_df.astype({"Z2.5": float})
    site_df.loc[:, "Z2.5"] /= 1000.0

    return site_df, station_df


def add_site_basins(site_df: pd.DataFrame, nzcvm_data_ffp: Path) -> pd.DataFrame:
    """
    Add the site basins to the site table

    Parameters
    ----------
    site_df : pd.DataFrame
        The site table dataframe with at least the columns 'lon' and 'lat'
        Ideally and in most cases, this dataframe should be the output of create_site_table_response
    nzcvm_data_ffp : Path
        The full file path to the nzcvm_data repository that stores the basin information

    Returns
    -------
    pd.DataFrame
        The site dataframe with the 'basin' column added
    """
    # Get the site table and points
    ll_points = site_df[["lon", "lat"]].values
    site_df["basin"] = None

    # Get the NZCVM version
    config = cfg.Config()
    nzcvm_version = config.get_value("nzcvm_version")
    priority_basins = config.get_value("priority_basins")

    # Create the CVMRegistry object
    registry_path = nzcvm_data_ffp / "nzcvm_registry.yaml"
    cvm_registry = registry.CVMRegistry(nzcvm_version, nzcvm_data_ffp, registry_path)

    # Make a new basin_dist from the registry
    basin_dict = {
        basin["name"].split("_")[0]: basin["boundaries"]
        for basin in cvm_registry.registry["basin"]
    }

    for basin in cvm_registry.global_params["basins"]:
        # Get the basin name
        basin_name = basin.split("_")[0]

        # Get the boundaries
        boundaries = basin_dict[basin_name]
        for boundary in boundaries:
            basin_outline = cvm_registry.load_basin_boundary(boundary)

            # Find sites within basin
            is_inside_basin = point_in_polygon.is_inside_postgis_parallel(
                ll_points, basin_outline
            )
            # Ensure we only update the basin of a site if it either doesn't have a basin or is in a priority basin
            mask_has_basin = site_df["basin"].notna()
            mask_priority = mask_has_basin & (basin_name in priority_basins)
            mask_no_basin = ~mask_has_basin
            mask_update = (
                is_inside_basin & mask_no_basin | mask_priority & is_inside_basin
            )
            site_df.loc[mask_update, "basin"] = basin_name

    # Add the nzcvm_version column
    site_df["nzcvm_version"] = nzcvm_version

    return site_df
