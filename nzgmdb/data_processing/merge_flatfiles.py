from pathlib import Path

import pandas as pd
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure


def merge_im_data(
    im_dir: Path,
    output_dir: Path,
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
    output_dir : Path
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
            "fmax_000": "fmax_mean_X",
            "fmax_090": "fmax_mean_Y",
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
            "CAV5",
            "AI",
            "Ds575",
            "Ds595",
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
        output_dir / file_structure.PreFlatfileNames.GROUND_MOTION_IM_CATALOGUE,
        index=False,
    )


def merge_flatfiles(main_dir: Path, bypass_records_ffp: Path = None):
    """
    Merge the flatfiles into the final flatfiles, separating the components
    and ensuring that the data contains only the unique events and sites that made it to the IM calculation

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    bypass_records_ffp : Path
        The full file path to the bypass records file, which includes a custom fmin, fmax, and p_wave_ix
    """
    # Get the flatfile directory
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Load the files
    event_df = pd.read_csv(
        flatfile_dir
        / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_AFTERSHOCKS,
        dtype={"evid": str},
    )
    sta_mag_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.STATION_MAGNITUDE_TABLE_GEONET,
        dtype={"evid": str},
    )
    phase_table_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
    )
    prop_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.PROPAGATION_TABLE,
        dtype={"evid": str},
    )
    im_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.GROUND_MOTION_IM_CATALOGUE,
        dtype={"loc": str, "evid": str},
    )
    site_basin_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.SITE_TABLE
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

    phase_table_df = phase_table_df[
        phase_table_df["record_id"].isin(im_df["record_id"])
    ]

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
                "aftershock_flag_crjb0",
                "cluster_flag_crjb0",
                "aftershock_flag_crjb2",
                "cluster_flag_crjb2",
                "aftershock_flag_crjb5",
                "cluster_flag_crjb5",
                "aftershock_flag_crjb10",
                "cluster_flag_crjb10",
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

    # Find the station location information with the inventory lat, lon and elev
    config = cfg.Config()
    channel_codes = ",".join(config.get_value("channel_codes"))
    client_NZ = FDSN_Client("GEONET")
    inventory = client_NZ.get_stations(channel=channel_codes, level="response")
    station_info = [
        [
            station.code,
            station.latitude,
            station.longitude,
            station.elevation,
        ]
        for network in inventory
        for station in network
    ]
    station_df = pd.DataFrame(
        station_info, columns=["sta", "sta_lat", "sta_lon", "sta_elev"]
    )

    # Merge the station information into the gm_im_df_flat
    gm_im_df_flat = gm_im_df_flat.merge(
        station_df[["sta", "sta_lat", "sta_lon", "sta_elev"]],
        on="sta",
        how="left",
        suffixes=("", "_new"),
    )

    # Find where sta_lat is nan and replace with the inventorys lat, lon and elev
    gm_im_df_flat["sta_lat"] = gm_im_df_flat["sta_lat"].fillna(
        gm_im_df_flat["sta_lat_new"]
    )
    gm_im_df_flat["sta_lon"] = gm_im_df_flat["sta_lon"].fillna(
        gm_im_df_flat["sta_lon_new"]
    )
    gm_im_df_flat["sta_elev"] = gm_im_df_flat["sta_elev"].fillna(
        gm_im_df_flat["sta_elev_new"]
    )

    # Drop the new columns
    gm_im_df_flat = gm_im_df_flat.drop(
        columns=["sta_lat_new", "sta_lon_new", "sta_elev_new"]
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

    # Merge in the bypass information
    if bypass_records_ffp is not None:
        bypass_df = pd.read_csv(bypass_records_ffp)
        gm_im_df_flat = gm_im_df_flat.merge(
            bypass_df[
                [
                    "record_id",
                    "fmax_000",
                    "fmax_090",
                    "fmax_ver",
                    "fmin_000",
                    "fmin_090",
                    "fmin_ver",
                ]
            ],
            on="record_id",
            how="left",
            suffixes=("", "_bypass"),
        )
        for bypass_col, col in [
            ("fmin_000", "fmin_X"),
            ("fmin_090", "fmin_Y"),
            ("fmin_ver", "fmin_Z"),
            ("fmax_000", "fmax_X"),
            ("fmax_090", "fmax_Y"),
            ("fmax_ver", "fmax_Z"),
        ]:
            gm_im_df_flat[col] = gm_im_df_flat[col].fillna(gm_im_df_flat[bypass_col])
        gm_im_df_flat = gm_im_df_flat.drop(
            columns=[
                "fmax_000",
                "fmax_090",
                "fmax_ver",
                "fmin_000",
                "fmin_090",
                "fmin_ver",
            ]
        )

        # Add any extra p_wave_ix values to the phase_table_df
        new_records = bypass_df[
            ~bypass_df["record_id"].isin(phase_table_df["record_id"])
        ]
        # remove p_wave_ix of nan
        new_records = new_records.dropna(subset=["p_wave_ix"])
        new_records = new_records[["record_id", "p_wave_ix"]]
        phase_table_df = pd.concat([phase_table_df, new_records])

    # Add in the default fmin values if they are nan
    default_fmin = config.get_value("low_cut_default")
    for col in ["fmin_X", "fmin_Y", "fmin_Z"]:
        gm_im_df_flat[col] = gm_im_df_flat[col].fillna(default_fmin)

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
            "CAV5",
            "AI",
            "Ds575",
            "Ds595",
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
            "aftershock_flag_crjb0",
            "cluster_flag_crjb0",
            "aftershock_flag_crjb2",
            "cluster_flag_crjb2",
            "aftershock_flag_crjb5",
            "cluster_flag_crjb5",
            "aftershock_flag_crjb10",
            "cluster_flag_crjb10",
        ]
        + psa_columns
        + fas_columns
    )
    gm_im_df_flat = gm_im_df_flat[columns]

    # Separate into different component files
    (
        df_000_flat,
        df_090_flat,
        df_ver_flat,
        df_geomean_flat,
        df_rotd0_flat,
        df_rotd50_flat,
        df_rotd100_flat,
        df_eas_flat,
    ) = (
        gm_im_df_flat[gm_im_df_flat.component == "000"],
        gm_im_df_flat[gm_im_df_flat.component == "090"],
        gm_im_df_flat[gm_im_df_flat.component == "ver"],
        gm_im_df_flat[gm_im_df_flat.component == "geom"],
        gm_im_df_flat[gm_im_df_flat.component == "rotd0"],
        gm_im_df_flat[gm_im_df_flat.component == "rotd50"],
        gm_im_df_flat[gm_im_df_flat.component == "rotd100"],
        gm_im_df_flat[gm_im_df_flat.component == "eas"],
    )

    # Remove NaN columns from the flatfiles with invalid components
    columns_remove_rotd = ["CAV", "CAV5", "AI", "Ds575", "Ds595"] + fas_columns
    columns_remove_eas = [
        "PGA",
        "PGV",
        "CAV",
        "CAV5",
        "AI",
        "Ds575",
        "Ds595",
    ] + psa_columns
    df_rotd0_flat = df_rotd0_flat.drop(columns=columns_remove_rotd)
    df_rotd50_flat = df_rotd50_flat.drop(columns=columns_remove_rotd)
    df_rotd100_flat = df_rotd100_flat.drop(columns=columns_remove_rotd)
    df_eas_flat = df_eas_flat.drop(columns=columns_remove_eas)

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
    df_geomean_flat.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_GEOM_FLAT,
        index=False,
    )
    df_rotd0_flat.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD0_FLAT,
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
    df_eas_flat.to_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_EAS_FLAT,
        index=False,
    )


def merge_dbs(
    flatfile_db_dir: Path,
    to_merge_db_dir: Path,
    output_dir: Path,
):
    """
    Merge the databases into a single database, where all the results from the to_merge_db_dir
    are adding or replacing the flatfile_db_dir results. The output is saved to the output_dir

    Parameters
    ----------
    flatfile_db_dir : Path
        The main database directory
    to_merge_db_dir : Path
        The directory of the database to merge into the main database
    output_dir : Path
        The directory to save the merged database
    """
    # For each file in the flatfiles, merge the to_merge_db_dir into the main_db_dir
    for flatfile_name in file_structure.FlatfileNames:
        main_df = pd.read_csv(flatfile_db_dir / flatfile_name, dtype={"evid": str})
        to_merge_df = pd.read_csv(to_merge_db_dir / flatfile_name, dtype={"evid": str})

        if flatfile_name == file_structure.FlatfileNames.EARTHQUAKE_SOURCE_TABLE:
            # Merge based on evid, replace values if they exist and append new ones
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["evid"], keep="last"
            )
            # Re-sort based on evid
            main_df = main_df.sort_values("datetime")
        elif flatfile_name == file_structure.FlatfileNames.STATION_MAGNITUDE_TABLE:
            # Make the unique record_id col with the columns evid_sta_chan_loc
            main_df["record_id"] = (
                main_df["evid"]
                + "_"
                + main_df["sta"]
                + "_"
                + main_df["chan"]
                + "_"
                + main_df["loc"].astype(str)
            )
            to_merge_df["record_id"] = (
                to_merge_df["evid"]
                + "_"
                + to_merge_df["sta"]
                + "_"
                + to_merge_df["chan"]
                + "_"
                + to_merge_df["loc"].astype(str)
            )
            # Merge on record_id, replace values if they exist and append new ones
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["record_id"], keep="last"
            )
            # Re-sort based on record_id
            main_df = main_df.sort_values("record_id")
            # Remove the record_id column
            main_df = main_df.drop(columns=["record_id"])
        elif flatfile_name == file_structure.FlatfileNames.SITE_TABLE:
            # Merge based on sta, replace values if they exist and append new ones
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["sta"], keep="last"
            )
            # Re-sort based on sta
            main_df = main_df.sort_values("sta")
        elif flatfile_name == file_structure.FlatfileNames.PROPAGATION_TABLE:
            # Merge based on evid_sta, replace values if they exist and append new ones
            main_df["evid_sta"] = main_df["evid"] + "_" + main_df["sta"].astype(str)
            to_merge_df["evid_sta"] = (
                to_merge_df["evid"] + "_" + to_merge_df["sta"].astype(str)
            )
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["evid_sta"], keep="last"
            )
            # Re-sort based on evid_sta
            main_df = main_df.sort_values("evid_sta")
            # Remove the record_id column
            main_df = main_df.drop(columns=["evid_sta"])
        else:
            # Merge on record_id, replace values if they exist and append new ones
            main_df = pd.concat([main_df, to_merge_df]).drop_duplicates(
                subset=["record_id"], keep="last"
            )
            # If the name of the file contains "flat" sort by datetime ,sta
            if "flat" in flatfile_name:
                main_df = main_df.sort_values(["datetime", "sta"])
            else:
                # Re-sort based on record_id
                main_df = main_df.sort_values("record_id")

        # Save the merged database
        main_df.to_csv(output_dir / flatfile_name, index=False)
