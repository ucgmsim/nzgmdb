from pathlib import Path

import pandas as pd

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure


def merge_im_data(
    im_dir: Path,
    ouptut_dir: Path,
    gmc_ffp: Path,
    fmax_ffp: Path,
):
    """
    Merge the IM data into a single flatfile. Also merges in the GMC and fmax data and
    filters out records that are below the Ds595 lower bound
    and then saves the skipped records to a separate file.

    Parameters
    ----------
    im_dir : Path
        The directory where the IM files are stored
    ouptut_dir : Path
        The directory to save the final IM flatfile and the skipped records
    gmc_ffp : Path
        The file path to the GMC results
    fmax_ffp : Path
        The file path to the fmax results
    """
    # Load the GMC file
    gmc_results = pd.read_csv(gmc_ffp)

    try:
        # Load the fmax file
        fmax_results = pd.read_csv(fmax_ffp)
    except pd.errors.EmptyDataError:
        fmax_results = pd.DataFrame(
            columns=["record_id", "fmax_000", "fmax_090", "fmax_ver"]
        )

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

    # Create a skipped IM merge file for records that are below the Ds595 lower bound
    Ds595_filter_records["reason"] = f"Ds595 below lower bound of {Ds595_lower_bound}"
    Ds595_filter_records.to_csv(
        ouptut_dir / file_structure.SkippedRecordFilenames.IM_MERGE_SKIPPED_RECORDS,
        index=False,
    )

    # Merge in fmax
    gm_final = pd.merge(
        gm_final,
        fmax_results,
        left_on="record_id",
        right_on="record_id",
        how="left",
    )
    # Rename fmax columns
    gm_final = gm_final.rename(
        columns={
            "fmax_000": "fmax_mean_Y",
            "fmax_090": "fmax_mean_X",
            "fmax_ver": "fmax_mean_Z",
        }
    )

    # Sort columns nicely
    psa_columns = gm_final.columns[gm_final.columns.str.contains("pSA")].tolist()
    fas_columns = gm_final.columns[gm_final.columns.str.contains("FAS")].tolist()
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
            "fmin_mean_X",
            "fmax_mean_X",
            "multi_mean_X",
            "score_mean_Y",
            "fmin_mean_Y",
            "fmax_mean_Y",
            "multi_mean_Y",
            "score_mean_Z",
            "fmin_mean_Z",
            "fmax_mean_Z",
            "multi_mean_Z",
        ]
        + psa_columns
        + fas_columns
    ]

    # Save the ground_motion_im_catalogue.csv
    gm_final.to_csv(
        ouptut_dir / file_structure.PreFlatfileNames.GROUND_MOTION_IM_CATALOGUE,
        index=False,
    )


def seperate_components(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Separate the components into the different component dataframes

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to separate, merged large gm_flatfile

    Returns
    -------
    df_000 : pd.DataFrame
        The 000 component dataframe
    df_090 : pd.DataFrame
        The 090 component dataframe
    df_ver : pd.DataFrame
        The ver component dataframe
    df_rotd50 : pd.DataFrame
        The rotd50 component dataframe
    df_rotd100 : pd.DataFrame
        The rotd100 component dataframe
    """
    df_000 = df[df.component == "000"]
    df_090 = df[df.component == "090"]
    df_ver = df[df.component == "ver"]
    df_rotd50 = df[df.component == "rotd50"]
    df_rotd100 = df[df.component == "rotd100"]
    return df_000, df_090, df_ver, df_rotd50, df_rotd100


def merge_flatfiles(main_dir: Path):
    """
    Merge the flatfiles into the final flatfiles, separating the components
    and ensuring that the data contains only the unique events and sites that made it to the IM calculation

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    """
    # Get the flatfile directory
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Load the files
    event_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_DISTANCES
    )
    sta_mag_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.STATION_MAGNITUDE_TABLE_GEONET
    )
    phase_table_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
    )
    prop_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.PROPAGATION_TABLE
    )
    im_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.GROUND_MOTION_IM_CATALOGUE,
        dtype={"loc": str},
    )
    site_basin_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.SITE_TABLE
    )
    plane_data_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.FAULT_PLANE_TABLE
    )

    # Get the recorders information for location codes
    config = cfg.Config()
    locations_url = config.get_value("locations_url")
    locations_df = pd.read_csv(locations_url)
    # Ensure the Station and Location pairings are unique
    locations_df = locations_df.drop_duplicates(subset=["Station", "Location"])

    # Ensure correct strike and rake values
    event_df.loc[event_df.strike == 360, "strike"] = 0
    event_df.loc[event_df.rake > 180, "rake"] -= 360

    # Get unique events that made it to the IM calculation
    unique_events = im_df.evid.unique()
    # Ensure that the other dfs only have the unique events
    event_df = event_df[event_df.evid.isin(unique_events)]
    phase_table_df = phase_table_df[phase_table_df.evid.isin(unique_events)]
    plane_data_df = plane_data_df[plane_data_df.evid.isin(unique_events)]

    # Ensure that the site_basin_df only has the unique sites found in the im_df
    unique_sites = im_df["sta"].unique()
    site_basin_df = site_basin_df[site_basin_df["sta"].isin(unique_sites)]

    # Ensure the station magnitude table only has values of events and station pairs available in the im_df
    unique_pairs_df = im_df[["evid", "sta"]].drop_duplicates()
    sta_mag_df = pd.merge(sta_mag_df, unique_pairs_df, on=["evid", "sta"], how="inner")

    # Get a list of sites not found in the site basin df
    missing_sites = set(unique_sites) - set(site_basin_df["sta"].unique())
    # Save the missing sites
    missing_sites_df = pd.DataFrame(missing_sites, columns=["sta"])
    missing_sites_df.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.MISSING_SITES, index=False
    )

    # Rename all the gmc column names to remove the middle _mean
    im_df = im_df.rename(
        columns={
            "score_mean_X": "score_X",
            "fmin_mean_X": "fmin_X",
            "fmax_mean_X": "fmax_X",
            "multi_mean_X": "multi_X",
            "score_mean_Y": "score_Y",
            "fmin_mean_Y": "fmin_Y",
            "fmax_mean_Y": "fmax_Y",
            "multi_mean_Y": "multi_Y",
            "score_mean_Z": "score_Z",
            "fmin_mean_Z": "fmin_Z",
            "fmax_mean_Z": "fmax_Z",
            "multi_mean_Z": "multi_Z",
        }
    )

    # Merge event data with the IM data
    gm_im_df_flat = im_df.merge(
        event_df[
            [
                "evid",
                "datetime",
                "lat",
                "lon",
                "depth",
                "mag",
                "mag_type",
                "tect_class",
                "reloc",
                "domain_no",
                "domain_type",
                "strike",
                "dip",
                "rake",
                "f_length",
                "f_width",
                "f_type",
                "z_tor",
                "z_bor",
            ]
        ],
        on="evid",
        how="left",
    )
    gm_im_df_flat = gm_im_df_flat.rename(
        columns={"lat": "ev_lat", "lon": "ev_lon", "depth": "ev_depth"}
    )

    # Merge in the site data
    gm_im_df_flat = gm_im_df_flat.merge(
        site_basin_df[
            [
                "sta",
                "lat",
                "lon",
                "elev",
                "Vs30",
                "Vs30_std",
                "Q_Vs30",
                "T0",
                "T0_std",
                "Q_T0",
                "Z1.0",
                "Z1.0_std",
                "Q_Z1.0",
                "Z2.5",
                "Z2.5_std",
                "Q_Z2.5",
                "site_domain_no",
            ]
        ],
        on="sta",
        how="left",
    )
    gm_im_df_flat = gm_im_df_flat.rename(
        columns={"lat": "sta_lat", "lon": "sta_lon", "elev": "sta_elev"}
    )

    # Merge in the location codes extra depth information where the station and location line up
    # locations_df has the column "Station" and "Location" and "Depth"
    gm_im_df_flat = (
        gm_im_df_flat.merge(
            locations_df[["Station", "Location", "Depth"]],
            left_on=["sta", "loc"],
            right_on=["Station", "Location"],
            how="left",
        )
        .drop(columns=["Station", "Location"])
        .rename(columns={"Depth": "loc_elev"})
    )

    # Flip the sign for the location elevation as it previously was depth
    gm_im_df_flat["loc_elev"] = -gm_im_df_flat["loc_elev"]

    # Add in a flag for when the location elevation is 0
    # Group by 'evid', 'sta', and 'chan'
    grouped = gm_im_df_flat.groupby(["evid", "sta", "chan"])

    # Custom function to handle NaN values and find the index of the row with the loc_elev value closest to 0
    def custom_idxmin(group: pd.DataFrame):
        # Filter out loc_elev values greater than 5 meters (In either direction)
        group = group[group["loc_elev"].abs() <= config.get_value("locations_max_elev")]
        if group["loc_elev"].isna().all():
            return None
        # Find the index of the row with the loc_elev value closest to 0
        return (group["loc_elev"].abs()).idxmin(skipna=True)

    # Find the index of the row with the smallest loc_elev value for each group, excluding NaN values
    idx_min_loc_elev = grouped.apply(custom_idxmin)

    gm_im_df_flat["is_ground_level"] = False
    # Set the flag to True for the rows with the smallest loc_elev value
    record_ids = gm_im_df_flat.loc[idx_min_loc_elev.dropna(), "record_id"]
    gm_im_df_flat.loc[
        gm_im_df_flat["record_id"].isin(record_ids), "is_ground_level"
    ] = True

    # For Locations not found in the dataframe, set the loc_elev to 0 only if there is just 1 location
    # Also set the is_ground_level to True
    gm_im_df_flat.loc[
        gm_im_df_flat["loc_elev"].isna()
        & gm_im_df_flat.groupby(["evid", "sta", "chan"])["loc"]
        .transform("nunique")
        .eq(1),
        ["is_ground_level", "loc_elev"],
    ] = [True, 0.0]
    # Replace -0.0 with 0.0 in the DataFrame
    gm_im_df_flat = gm_im_df_flat.replace(-0.0, 0.0)

    # Remove duplicated columns in prop_df
    prop_df["evid_sta"] = prop_df["evid"].astype(str) + "_" + prop_df["sta"].astype(str)
    prop_df = prop_df.drop_duplicates(subset=["evid_sta"])
    prop_df = prop_df.drop(columns=["evid_sta"])

    # Merge in the propagation data
    gm_im_df_flat = gm_im_df_flat.merge(
        prop_df[
            [
                "evid",
                "sta",
                "r_epi",
                "r_hyp",
                "r_jb",
                "r_rup",
                "r_x",
                "r_y",
                "r_tvz",
                "r_xvf",
            ]
        ],
        on=["evid", "sta"],
        how="left",
    )

    # Add in colunms for fmin_max and fmin_highpass
    gm_im_df_flat["fmin_max"] = gm_im_df_flat[["fmin_X", "fmin_Y", "fmin_Z"]].apply(
        max, axis=1
    )
    gm_im_df_flat["HPF"] = gm_im_df_flat["fmin_max"] / 1.25

    # Sort the rows
    gm_im_df_flat = gm_im_df_flat.sort_values(["datetime", "sta", "component"])

    # Re-sort the columns
    psa_columns = gm_im_df_flat.columns[
        gm_im_df_flat.columns.str.contains("pSA")
    ].tolist()
    fas_columns = gm_im_df_flat.columns[
        gm_im_df_flat.columns.str.contains("FAS")
    ].tolist()
    columns = (
        [
            "record_id",
            "datetime",
            "evid",
            "sta",
            "loc",
            "chan",
            "component",
            "ev_lat",
            "ev_lon",
            "ev_depth",
            "mag",
            "mag_type",
            "tect_class",
            "reloc",
            "domain_no",
            "domain_type",
            "strike",
            "dip",
            "rake",
            "f_length",
            "f_width",
            "f_type",
            "z_tor",
            "z_bor",
            "sta_lat",
            "sta_lon",
            "sta_elev",
            "loc_elev",
            "is_ground_level",
            "r_epi",
            "r_hyp",
            "r_jb",
            "r_rup",
            "r_x",
            "r_y",
            "r_tvz",
            "r_xvf",
            "Vs30",
            "Vs30_std",
            "Q_Vs30",
            "T0",
            "T0_std",
            "Q_T0",
            "Z1.0",
            "Z1.0_std",
            "Q_Z1.0",
            "Z2.5",
            "Z2.5_std",
            "Q_Z2.5",
            "site_domain_no",
            "PGA",
            "PGV",
            "CAV",
            "AI",
            "Ds575",
            "Ds595",
            "MMI",
            "score_X",
            "fmin_X",
            "fmax_X",
            "multi_X",
            "score_Y",
            "fmin_Y",
            "fmax_Y",
            "multi_Y",
            "score_Z",
            "fmin_Z",
            "fmax_Z",
            "multi_Z",
            "fmin_max",
            "HPF",
        ]
        + psa_columns
        + fas_columns
    )
    gm_im_df_flat = gm_im_df_flat[columns]

    # Separate into different components for both flatfile and normal im data
    df_000, df_090, df_ver, df_rotd50, df_rotd100 = seperate_components(im_df)
    (
        df_000_flat,
        df_090_flat,
        df_ver_flat,
        df_rotd50_flat,
        df_rotd100_flat,
    ) = seperate_components(gm_im_df_flat)

    # Save final outputs
    event_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.EARTHQUAKE_SOURCE_TABLE, index=False
    )
    sta_mag_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.STATION_MAGNITUDE_TABLE, index=False
    )
    phase_table_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.PHASE_ARRIVAL_TABLE, index=False
    )
    site_basin_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.SITE_TABLE, index=False
    )
    prop_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.PROPAGATION_TABLE, index=False
    )
    df_000.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_000, index=False
    )
    df_090.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_090, index=False
    )
    df_ver.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_VER, index=False
    )
    df_rotd50.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD50, index=False
    )
    df_rotd100.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD100,
        index=False,
    )
    df_000_flat.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_000_FLAT,
        index=False,
    )
    df_090_flat.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_090_FLAT,
        index=False,
    )
    df_ver_flat.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_VER_FLAT,
        index=False,
    )
    df_rotd50_flat.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD50_FLAT,
        index=False,
    )
    df_rotd100_flat.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD100_FLAT,
        index=False,
    )
    plane_data_df.to_csv(
        flatfile_dir / file_structure.FlatfileNames.FAULT_PLANE_TABLE, index=False
    )
