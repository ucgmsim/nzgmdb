from pathlib import Path

import pandas as pd

from nzgmdb.management import config as cfg, file_structure


def create_quality_db(
    main_dir: Path,
    fmax_df: pd.DataFrame = None,
    bypass_records_ffp: Path = None,
    n_procs: int = 1,
):
    """
    Create the quality database by running the following checks:
    1) Check for Ds595 < 3s
    2) Check against GMC predictions score mean
    3) Check against GMC predictions multi mean

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    fmax_df : pd.DataFrame, optional
        The dataframe containing the fmax values, by default None (will grab default fmax_df from flatfiles)
    bypass_records_ffp : Path, optional
        The file path to the records that will bypass the quality checks
    n_procs : int, optional
        Number of processes to use, by default 1
    """
    # Make the quality db directory
    output_dir = main_dir / "quality_db"
    output_dir.mkdir(exist_ok=True)

    # Load the ground motion im catalog
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    gm_df = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.GROUND_MOTION_IM_CATALOGUE,
        dtype={"evid": str},
    )

    # Get the Ds595 Lower Bound
    config = cfg.Config()
    Ds595_lower_bound = config.get_value("Ds595_lower_bound")

    # Filter the Ds595 by first filtering only the ver, 000 and 090 components
    comp_sub = gm_df[gm_df.component.isin(["ver", "000", "090"])]
    # Then Sum the Ds595 values for each record
    comp_sub_grouped = comp_sub.groupby(["record_id"]).sum()
    # Find the records that are below the Ds595 lower bound
    Ds595_filter_records = comp_sub_grouped[
        comp_sub_grouped.Ds595 < Ds595_lower_bound
    ].reset_index()[["record_id"]]

    # Create a skipped IM merge file for records that are below the Ds595 lower bound
    Ds595_filter_records["reason"] = f"Ds595 below lower bound of {Ds595_lower_bound}"
    Ds595_filter_records.to_csv(
        output_dir / file_structure.SkippedRecordFilenames.IM_MERGE_SKIPPED_RECORDS,
        index=False,
    )

    # Remove the records that are below the Ds595 lower bound
    gm_ds595 = gm_df[~gm_df.record_id.isin(Ds595_filter_records.record_id)]

    config = cfg.Config()
    fmax_min = config.get_value("fmax_min")
    score_min = config.get_value("score_min")
    fmin_max = config.get_value("fmin_max")
    multi_max = config.get_value("multi_max")

    fmax_df = (
        pd.read_csv(
            flatfile_dir / file_structure.FlatfileNames.FMAX,
            dtype={"record_id": str},
        )
        if fmax_df is None
        else fmax_df
    )

    # Merge fmax into the gm_ds595 dataframe
    gm_fmax_df = gm_ds595.merge(fmax_df, on="record_id", how="left")

    # Find fmin_max and fmax_min
    gm_fmax_df["fmin_max"] = gm_fmax_df[
        ["fmin_mean_X", "fmin_mean_Y", "fmin_mean_Z"]
    ].apply(max, axis=1)
    gm_fmax_df["fmax_min"] = gm_fmax_df[["fmax_000", "fmax_090", "fmax_ver"]].apply(
        min, axis=1
    )

    # Fill nan values with 1 / (2.5 * dt)

    # Filter out records that have too low of a fmax_min value
    fmax_min_filter = gm_fmax_df[gm_fmax_df["fmax_min"] < fmax_min]

    # if fmax is not None and fmax <= fmax_min:
    #     skipped_record_dict = {
    #         "record_id": mseed_stem,
    #         "reason": f"Fmax value is less than {fmax_min}",
    #     }
    #     return pd.DataFrame([skipped_record_dict])
    #
    # # Filter by score, fmin and multi mean
    # if gmc_rows["score_mean"].min() < score_min:
    #     skipped_record_dict = {
    #         "record_id": mseed_stem,
    #         "reason": f"Score mean is less than {score_min}",
    #     }
    #     return pd.DataFrame([skipped_record_dict])
    # if fmin > fmin_max:
    #     skipped_record_dict = {
    #         "record_id": mseed_stem,
    #         "reason": f"Fmin mean is greater than {fmin_max}",
    #     }
    #     return pd.DataFrame([skipped_record_dict])
    # if gmc_rows["multi_mean"].max() > multi_max:
    #     skipped_record_dict = {
    #         "record_id": mseed_stem,
    #         "reason": f"Multi mean is greater than {multi_max}",
    #     }
    #     return pd.DataFrame([skipped_record_dict])
