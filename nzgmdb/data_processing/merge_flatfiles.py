from pathlib import Path

import pandas as pd
import numpy as np

from nzgmdb.management import file_structure


def merge_im_data(
    main_dir: Path,
    gmc_ffp: Path,
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
    record_id_split = gm_final["record_id"].str.split("_", expand=True)
    gm_final[["evid", "sta", "chan", "loc"]] = gm_final["record_id"].str.split(
        "_", expand=True
    )

    # remove the record column
    gm_final = gm_final.drop(columns=["record"])

    # Save the ground_motion_im_catalogue.csv
    gm_final.to_csv(flatfile_dir / "ground_motion_im_catalogue.csv", index=False)


merge_im_data(
    main_dir=Path("/home/joel/local/gmdb/US_stuff/new_struct_2022"),
    gmc_ffp=Path(
        "/home/joel/local/gmdb/US_stuff/new_struct_2022/flatfiles/gmc_predictions.csv"
    ),
)
