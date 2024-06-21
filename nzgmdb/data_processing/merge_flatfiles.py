from pathlib import Path

import pandas as pd

from nzgmdb.management import file_structure, config as cfg


def merge_im_data(
    im_dir: Path,
    ouptut_dir: Path,
    gmc_ffp: Path,
    fmax_ffp: Path,
):
    """
    Merge the IM data into a single flatfile
    """
    # Get the flatfile directory
    # flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    # Get the IM directory
    # im_dir = file_structure.get_im_dir(main_dir)

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

    # Create a skipped IM merge file for records that are below the Ds595 lower bound
    Ds595_filter_records["reason"] = f"Ds595 below lower bound of {Ds595_lower_bound}"
    Ds595_filter_records.to_csv(
        ouptut_dir / "IM_merge_skipped_records.csv", index=False
    )

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
    gm_final.to_csv(ouptut_dir / "ground_motion_im_catalogue.csv", index=False)


def seperate_components(df: pd.DataFrame):
    """
    Seperate the components into the different components and remove columns that are
    not needed for each of the components
    """
    df_000 = df[df.component == "000"]
    df_090 = df[df.component == "090"]
    df_ver = df[df.component == "ver"]
    df_rotd50 = df[df.component == "rotd50"]
    df_rotd100 = df[df.component == "rotd100"]

    df_000 = df_000.drop(
        [
            "score_mean_X",
            "fmin_mean_X",
            "fmax_mean_X",
            "multi_mean_X",
            "score_mean_Z",
            "fmin_mean_Z",
            "fmax_mean_Z",
            "multi_mean_Z",
        ],
        axis=1,
    )
    df_090 = df_090.drop(
        [
            "score_mean_Y",
            "fmin_mean_Y",
            "fmax_mean_Y",
            "multi_mean_Y",
            "score_mean_Z",
            "fmin_mean_Z",
            "fmax_mean_Z",
            "multi_mean_Z",
        ],
        axis=1,
    )
    df_ver = df_ver.drop(
        [
            "score_mean_X",
            "fmin_mean_X",
            "fmax_mean_X",
            "multi_mean_X",
            "score_mean_Y",
            "fmin_mean_Y",
            "fmax_mean_Y",
            "multi_mean_Y",
        ],
        axis=1,
    )
    df_rotd50 = df_rotd50.drop(
        ["score_mean_Z", "fmin_mean_Z", "fmax_mean_Z", "multi_mean_Z"], axis=1
    )
    df_rotd50 = df_rotd50.drop(
        ["score_mean_Z", "fmin_mean_Z", "fmax_mean_Z", "multi_mean_Z"], axis=1
    )

    return df_000, df_090, df_ver, df_rotd50, df_rotd100


def merge_flatfiles(main_dir: Path):
    # Get the flatfile directory
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Load the files
    event_df = pd.read_csv(flatfile_dir / "earthquake_source_table_complete.csv")
    sta_mag_df = pd.read_csv(flatfile_dir / "station_magnitude_table.csv")
    phase_table_df = pd.read_csv(flatfile_dir / "phase_arrival_table.csv")
    prop_df = pd.read_csv(flatfile_dir / "propagation_path_table.csv")
    im_df = pd.read_csv(flatfile_dir / "ground_motion_im_catalogue.csv")
    site_basin_df = pd.read_csv(flatfile_dir / "site_table_basin.csv")

    # Ensure correct strike and rake values
    event_df.loc[event_df.strike == 360, "strike"] = 0
    event_df.loc[event_df.rake > 180, "rake"] -= 360

    # Get unique events that made it to the IM calculation
    unique_events = im_df.evid.unique()
    # Ensure that the other dfs only have the unique events
    event_df = event_df[event_df.evid.isin(unique_events)]
    sta_mag_df = sta_mag_df[
        sta_mag_df.evid.isin(unique_events)
    ]  # INCLUDE STATION FILTER TOO
    phase_table_df = phase_table_df[phase_table_df.evid.isin(unique_events)]

    # Ensure that the site_basin_df only has the unique sites found in the im_df
    unique_sites = im_df["sta"].unique()
    site_basin_df = site_basin_df[site_basin_df["sta"].isin(unique_sites)]

    # Get a list of sites not found in the site basin df
    missing_sites = set(unique_sites) - set(site_basin_df["sta"].unique())
    # Save the missing sites
    missing_sites_df = pd.DataFrame(missing_sites, columns=["sta"])
    missing_sites_df.to_csv(flatfile_dir / "missing_sites.csv", index=False)

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
    gm_im_df_flat = gm_im_df_flat.rename(columns={"lat": "sta_lat", "lon": "sta_lon"})

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
            "clip_prob",
            "clipped",
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
    event_df.to_csv(flatfile_dir / "earthquake_source_table.csv", index=False)
    sta_mag_df.to_csv(flatfile_dir / "station_magnitude_table.csv", index=False)
    phase_table_df.to_csv(flatfile_dir / "phase_arrival_table.csv", index=False)
    # TODO add site basin here
    df_000.to_csv(flatfile_dir / "ground_motion_im_table_000.csv", index=False)
    df_090.to_csv(flatfile_dir / "ground_motion_im_table_090.csv", index=False)
    df_ver.to_csv(flatfile_dir / "ground_motion_im_table_ver.csv", index=False)
    df_rotd50.to_csv(flatfile_dir / "ground_motion_im_table_rotd50.csv", index=False)
    df_rotd100.to_csv(flatfile_dir / "ground_motion_im_table_rotd100.csv", index=False)
    df_000_flat.to_csv(
        flatfile_dir / "ground_motion_im_table_000_flat.csv", index=False
    )
    df_090_flat.to_csv(
        flatfile_dir / "ground_motion_im_table_090_flat.csv", index=False
    )
    df_ver_flat.to_csv(
        flatfile_dir / "ground_motion_im_table_ver_flat.csv", index=False
    )
    df_rotd50_flat.to_csv(
        flatfile_dir / "ground_motion_im_table_rotd50_flat.csv", index=False
    )
    df_rotd100_flat.to_csv(
        flatfile_dir / "ground_motion_im_table_rotd100_flat.csv", index=False
    )
