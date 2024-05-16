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

    # Find all the gm_all files
    gm_all_files = im_dir.glob("**/**/gm_all.csv")

    # Concat all the gm_all files
    gm_all = pd.concat([pd.read_csv(file) for file in gm_all_files])

    # Merge the gm_all and new_df on record
    gm_final = pd.merge(
        gm_all,
        new_df,
        on="record",
        how="left",
    )

    # Add the chan, loc and rename event_id and station
    record_id_split = gm_final["record"].str.split("_")
    gm_final["evid"] = record_id_split[0]
    gm_final["sta"] = record_id_split[1]
    gm_final["chan"] = record_id_split[2]
    gm_final["loc"] = record_id_split[3].astype("int")

    # Save the ground_motion_im_catalogue.csv
    gm_final.to_csv(flatfile_dir / "ground_motion_im_catalogue.csv", index=False)


merge_im_data(
    main_dir=Path("/home/joel/local/gmdb/US_stuff/new_struct_2022"),
    gmc_ffp=Path(
        "/home/joel/local/gmdb/US_stuff/new_struct_2022/flatfiles/gmc_predictions.csv"
    ),
)
