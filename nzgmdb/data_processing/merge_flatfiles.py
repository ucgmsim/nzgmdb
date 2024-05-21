from pathlib import Path

import pandas as pd

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


merge_im_data(
    main_dir=Path("/home/joel/local/gmdb/US_stuff/new_struct_2022"),
    gmc_ffp=Path(
        "/home/joel/local/gmdb/US_stuff/new_struct_2022/flatfiles/gmc_predictions.csv"
    ),
    fmax_ffp=Path("/home/joel/local/gmdb/US_stuff/new_struct_2022/flatfiles/fmax.csv"),
)
