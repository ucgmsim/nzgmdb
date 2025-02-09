from pathlib import Path

import alphashape
import numpy as np
import pandas as pd
from shapely import Polygon

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
