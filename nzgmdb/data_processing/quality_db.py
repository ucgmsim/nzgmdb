from pathlib import Path

import numpy as np
import pandas as pd

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from nzgmdb.management.file_structure import FlatfileNames


def filter_flatfiles_on_catalouge(
    flatfile_dir: Path, final_output: Path, rotd50_flat: pd.DataFrame
):
    """
    Filter the flatfiles based on the records in the rotd50_flat dataframe.
    Ensures that the flatfiles are only for the records that are in the rotd50_flat dataframe.

    Parameters
    ----------
    flatfile_dir : Path
        The directory containing the flatfiles
    final_output : Path
        The directory to output the filtered flatfiles
    rotd50_flat : pd.DataFrame
        The dataframe containing the records to filter on
    """
    file_to_filter = [
        FlatfileNames.EARTHQUAKE_SOURCE_TABLE,
        FlatfileNames.FMAX,
        FlatfileNames.STATION_MAGNITUDE_TABLE,
        FlatfileNames.SITE_TABLE,
        FlatfileNames.PHASE_ARRIVAL_TABLE,
        FlatfileNames.PROPAGATION_TABLE,
        FlatfileNames.GMC_PREDICTIONS,
        FlatfileNames.SNR_METADATA,
        FlatfileNames.GROUND_MOTION_IM_000_FLAT,
        FlatfileNames.GROUND_MOTION_IM_090_FLAT,
        FlatfileNames.GROUND_MOTION_IM_VER_FLAT,
        FlatfileNames.GROUND_MOTION_IM_ROTD0_FLAT,
        FlatfileNames.GROUND_MOTION_IM_ROTD100_FLAT,
        FlatfileNames.GROUND_MOTION_IM_GEOM_FLAT,
        FlatfileNames.GROUND_MOTION_IM_EAS_FLAT,
    ]

    for file in file_to_filter:
        # Load the new file and filter based on record_id
        df = pd.read_csv(flatfile_dir / file, dtype={"evid": str})
        if file == FlatfileNames.EARTHQUAKE_SOURCE_TABLE:
            # filter by evid
            df_filtered = df[df["evid"].isin(rotd50_flat["evid"])]
        elif file == FlatfileNames.STATION_MAGNITUDE_TABLE:
            # Ensure loc is str
            df["loc"] = df["loc"].astype(str)
            # Make the record_id column
            df["record_id"] = (
                df["evid"]
                + "_"
                + df["sta"]
                + "_"
                + df["chan"].str[:2]
                + "_"
                + df["loc"]
            )
            df_filtered = df[df["record_id"].isin(rotd50_flat["record_id"])]
            # remove the record_id column
            df_filtered = df_filtered.drop(columns=["record_id"])
        elif file == FlatfileNames.SITE_TABLE:
            df_filtered = df[df["sta"].isin(rotd50_flat["sta"])]
        elif file == FlatfileNames.PROPAGATION_TABLE:
            # Make the evid_sta column
            df["evid_sta"] = df["evid"] + "_" + df["sta"]
            # Assert the same length of unique values
            assert len(df["evid_sta"].unique()) == len(df)
            # Create the rodtd50 evid_sta
            rotd50_flat["evid_sta"] = rotd50_flat["evid"] + "_" + rotd50_flat["sta"]
            df_filtered = df[df["evid_sta"].isin(rotd50_flat["evid_sta"])]
            # remove the evid_sta column
            df_filtered = df_filtered.drop(columns=["evid_sta"])
            rotd50_flat = rotd50_flat.drop(columns=["evid_sta"])
        elif file == FlatfileNames.GMC_PREDICTIONS:
            df_filtered = df[df["record"].isin(rotd50_flat["record_id"])]
        else:
            df_filtered = df[df["record_id"].isin(rotd50_flat["record_id"])]
        df_filtered.to_csv(final_output / file, index=False)


def filter_has_score_mean(catalog: pd.DataFrame, bypass_records: np.ndarray = None):
    """
    Filter the catalog based on if there is a score from GMC.

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality

    Returns
    -------
    pd.DataFrame
        The filtered catalog
    pd.DataFrame
        The skipped records
    """
    # Find records that do not have a score value (is same across all components)
    has_score_filter = catalog[catalog["score_X"].isna()]

    # Remove the bypass records if they exist
    if bypass_records is not None:
        has_score_filter = has_score_filter[
            ~has_score_filter["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from has_score_filter
    skipped_records = pd.DataFrame(
        {
            "record_id": has_score_filter["record_id"],
            "reason": "No score values from GMC",
        }
    )

    # Filter out the has_score records out of the catalog
    catalog = catalog[~catalog["record_id"].isin(has_score_filter["record_id"])]

    return catalog, skipped_records


def filter_score_mean(
    catalog: pd.DataFrame,
    score_min: float,
    bypass_records: np.ndarray = None,
    include_z: bool = False,
):
    """
    Filter the catalog based on the score_mean value from GMC.
    Only looks at X and Y components by default, can include Z.

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    score_min : float
        The minimum score value to filter on
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks
    include_z : bool, optional
        Whether to include the Z component, by default False

    Returns
    -------
    pd.DataFrame
        The filtered catalog
    pd.DataFrame
        The skipped records
    """
    # Find records that have too low of a score_X or score_Y value (or score_Z if include_z)
    score_min_filter = catalog[
        (
            (catalog["score_X"] < score_min)
            | (catalog["score_Y"] < score_min)
            | (catalog["score_Z"] < score_min)
            if include_z
            else (catalog["score_X"] < score_min) | (catalog["score_Y"] < score_min)
        )
    ]

    # Remove the bypass records if they exist
    if bypass_records is not None:
        score_min_filter = score_min_filter[
            ~score_min_filter["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from score_min_filter
    skipped_records = pd.DataFrame(
        {
            "record_id": score_min_filter["record_id"],
            "reason": f"Score mean is less than {score_min}",
        }
    )

    # Filter out the score_min records out of the catalog
    catalog = catalog[~catalog["record_id"].isin(score_min_filter["record_id"])]

    return catalog, skipped_records


def filter_multi_mean(
    catalog: pd.DataFrame,
    multi_max: float,
    bypass_records: np.ndarray = None,
    include_z: bool = False,
):
    """
    Filter the catalog based on the multi_mean value
    Only looks at X and Y components by default, can include Z.

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    multi_max : float
        The maximum multi_mean value to filter on
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks
    include_z : bool, optional
        Whether to include the Z component, by default False

    Returns
    -------
    pd.DataFrame
        The filtered catalog
    pd.DataFrame
        The skipped records
    """
    # Find records that have too high of a multi_X or multi_Y or multi_Z value
    multi_max_filter = catalog[
        (
            (catalog["multi_X"] > multi_max)
            | (catalog["multi_Y"] > multi_max)
            | (catalog["multi_Z"] > multi_max)
            if include_z
            else ((catalog["multi_X"] > multi_max) | (catalog["multi_Y"] > multi_max))
        )
    ]

    # Remove the bypass records if they exist
    if bypass_records is not None:
        multi_max_filter = multi_max_filter[
            ~multi_max_filter["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from multi_max_filter
    skipped_records = pd.DataFrame(
        {
            "record_id": multi_max_filter["record_id"],
            "reason": f"Multi mean is greater than {multi_max}",
        }
    )

    # Filter out the multi_max records out of the catalog
    catalog = catalog[~catalog["record_id"].isin(multi_max_filter["record_id"])]

    return catalog, skipped_records


def filter_fmax(
    catalog: pd.DataFrame, fmax_min: float, bypass_records: np.ndarray = None
):
    """
    Filter the catalog based on the fmax_min value

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    fmax_min : float
        The minimum fmax value to filter on
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks

    Returns
    -------
    pd.DataFrame
        The filtered catalog
    pd.DataFrame
        The skipped records
    """
    # Find fmax_min
    catalog.loc[:, "fmax_min"] = catalog[["fmax_X", "fmax_Y", "fmax_Z"]].apply(
        min, axis=1
    )

    # Find records that have too low of a fmax_min value
    fmax_min_filter = catalog[catalog["fmax_min"] < fmax_min]

    # Remove the bypass records if they exist
    if bypass_records is not None:
        fmax_min_filter = fmax_min_filter[
            ~fmax_min_filter["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from fmax_min_filter
    skipped_records = pd.DataFrame(
        {
            "record_id": fmax_min_filter["record_id"],
            "reason": f"Fmax value is less than {fmax_min}",
        }
    )

    # Filter out the fmax_min records out of the catalog
    catalog = catalog[~catalog["record_id"].isin(fmax_min_filter["record_id"])]

    # Remove the fmax_min column
    catalog = catalog.drop(columns=["fmax_min"])

    return catalog, skipped_records


def filter_fmin(
    catalog: pd.DataFrame, fmin_max: float, bypass_records: np.ndarray = None
):
    """
    Filter the catalog based on the fmin_max value

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    fmin_max : float
        The maximum fmin value to filter on
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks

    Returns
    -------
    pd.DataFrame
        The filtered catalog
    pd.DataFrame
        The skipped records
    """
    # Find records that have too high of a fmin_max value
    fmin_max_filter = catalog[catalog["fmin_max"] > fmin_max]

    # Remove the bypass records if they exist
    if bypass_records is not None:
        fmin_max_filter = fmin_max_filter[
            ~fmin_max_filter["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from fmin_max_filter
    skipped_records = pd.DataFrame(
        {
            "record_id": fmin_max_filter["record_id"],
            "reason": f"Fmin value is greater than {fmin_max}",
        }
    )

    # Filter out the fmin_max records out of the catalog
    catalog = catalog[~catalog["record_id"].isin(fmin_max_filter["record_id"])]

    return catalog, skipped_records


def filter_missing_sta_info(catalog: pd.DataFrame, bypass_records: np.ndarray = None):
    """
    Filter the catalog based on the missing station information

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks

    Returns
    -------
    pd.DataFrame
        The filtered catalog
    pd.DataFrame
        The skipped records
    """
    # Find records that are missing station information
    missing_sta_filter = catalog[catalog["Vs30"].isna()]

    # Remove the bypass records if they exist
    if bypass_records is not None:
        missing_sta_filter = missing_sta_filter[
            ~missing_sta_filter["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from missing_sta_filter
    skipped_records = pd.DataFrame(
        {
            "record_id": missing_sta_filter["record_id"],
            "reason": "Missing station information",
        }
    )

    # Filter out the missing_sta records out of the catalog
    catalog = catalog[~catalog["record_id"].isin(missing_sta_filter["record_id"])]

    return catalog, skipped_records


def filter_ground_level_locations(
    catalog: pd.DataFrame, bypass_records: np.ndarray = None
):
    """
    Filter the catalog based on the ground level locations

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks


    Returns
    -------
    pd.DataFrame
        The filtered catalog
    pd.DataFrame
        The skipped records
    """
    # Filter records that are not ground level
    ground_level_filter = catalog[~catalog["is_ground_level"]]

    # Remove the bypass records if they exist
    if bypass_records is not None:
        ground_level_filter = ground_level_filter[
            ~ground_level_filter["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from ground_level_filter
    skipped_records = pd.DataFrame(
        {
            "record_id": ground_level_filter["record_id"],
            "reason": "Not ground level location",
        }
    )

    # Filter out the non ground_level records out of the catalog
    catalog = catalog[~catalog["record_id"].isin(ground_level_filter["record_id"])]

    return catalog, skipped_records


def apply_clipNet_filter(
    catalog: pd.DataFrame,
    clipped_records_ffp: Path,
    bypass_records: np.ndarray = None,
):
    """
    Apply the ClipNet filter to the catalog
    Removes the clipped records from the catalog and creates a skipped records dataframe

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    clipped_records_ffp : Path
        The file path to the clipped records (created during the GeoNet processing)
    bypass_records : np.ndarray, optional
        The records to bypass the quality
    """
    # Read the clipped records
    try:
        clipped_records = pd.read_csv(clipped_records_ffp)
    except pd.errors.EmptyDataError:
        return catalog, pd.DataFrame(columns=["record_id", "reason"])

    # Remove the bypass records if they exist
    if bypass_records is not None:
        clipped_records = clipped_records[
            ~clipped_records["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from clipped_records
    skipped_records = pd.DataFrame(
        {
            "record_id": clipped_records["record_id"],
            "reason": "Clipped by ClipNet",
        }
    )

    # Filter out the clipped records out of the catalog
    catalog = catalog[~catalog["record_id"].isin(clipped_records["record_id"])]

    return catalog, skipped_records


def filter_duplicate_channels(catalog: pd.DataFrame, bypass_records: np.ndarray = None):
    """
    Filter the catalog based on the duplicate channels.
    Selects HN over BN except for the bypass records if the BN
    for the duplicate evid / sta is selected.

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks

    Returns
    -------
    pd.DataFrame
        The filtered catalog
    pd.DataFrame
        The skipped records
    """
    # Remove all channels that are not HN or BN
    catalog = catalog[catalog["chan"].isin(["HN", "BN"])]

    # Find same evid_sta combos by combining evid and sta columns
    catalog["evid_sta"] = catalog["evid"].astype(str) + "_" + catalog["sta"]

    # Get the ones that are duplicated
    dup_mask = catalog["evid_sta"].duplicated(keep=False)

    # Select all the BN ones from the original dataframe to remove that are duplicates
    duplicate_channels_filter = catalog.loc[dup_mask & (catalog["chan"] == "BN")]

    # Remove the bypass records if they exist and add other duplicated channels to ignore
    if bypass_records is not None:
        # Get the catalog records that are in the bypass_records
        bypass_records_mask = catalog.loc[
            catalog["record_id"].isin(bypass_records), "evid_sta"
        ]
        bypass_records_mask = catalog[catalog["evid_sta"].isin(bypass_records_mask)]

        # remove the bypass records from the bypass_records_mask
        add_to_duplicated = bypass_records_mask[
            ~bypass_records_mask["record_id"].isin(bypass_records)
        ]

        # Add the non bypass records to the duplicate_channels_filter that are duplicates
        duplicate_channels_filter = pd.concat(
            [
                duplicate_channels_filter,
                catalog.loc[catalog["record_id"].isin(add_to_duplicated["record_id"])],
            ]
        )

        # Remove the bypass records from the duplicate_channels_filter
        duplicate_channels_filter = duplicate_channels_filter[
            ~duplicate_channels_filter["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from duplicate_channels_filter
    skipped_records = pd.DataFrame(
        {
            "record_id": duplicate_channels_filter["record_id"],
            "reason": "Duplicate channels",
        }
    )

    # Filter out the duplicate channel records out of the catalog
    catalog = catalog[
        ~catalog["record_id"].isin(duplicate_channels_filter["record_id"])
    ]

    # Ensure that there is no duplictes in the evid_sta column
    assert len(catalog["evid_sta"].unique()) == len(catalog)

    # Remove the evid_sta column
    catalog = catalog.drop(columns=["evid_sta"])

    return catalog, skipped_records


def apply_all_filters(
    catalog: pd.DataFrame,
    clipped_records_ffp: Path,
    bypass_records: np.ndarray = None,
    score_min: float = None,
    multi_max: float = None,
    fmax_min: float = None,
    fmin_max: float = None,
):
    """
    Apply all the quality filters to the catalog
    Does the following:
    1) Filter by contains GMC predictions
    2) Filter by score mean
    3) Filter by multi mean
    4) Filter by fmax
    5) Filter by fmin
    6) Filter by missing station information
    7) Ensure we use ground level locations
    8) Filter out clipped records
    9) Select which channel to use for duplicate HN, BN for the same evid / sta

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog dataframe to filter
    clipped_records_ffp : Path
        The file path to the clipped records (created during the GeoNet processing)
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks
    score_min: float, optional
        The minimum score value to filter on
    multi_max: float, optional
        The maximum multi_mean value to filter on
    fmax_min: float, optional
        The minimum fmax value to filter on
    fmin_max: float, optional
        The maximum fmin value to filter on

    Returns
    -------
    pd.DataFrame
        The filtered catalog
    pd.DataFrame
        The skipped records
    """
    config = cfg.Config()

    # Get the config values if they are not provided
    score_min = score_min if score_min is not None else config.get_value("score_min")
    multi_max = multi_max if multi_max is not None else config.get_value("multi_max")
    fmax_min = fmax_min if fmax_min is not None else config.get_value("fmax_min")
    fmin_max = fmin_max if fmin_max is not None else config.get_value("fmin_max")

    # Filter by has score mean
    catalog, skipped_records_has_score = filter_has_score_mean(catalog, bypass_records)

    # Filter by score mean
    catalog, skipped_records_score = filter_score_mean(
        catalog, score_min, bypass_records
    )

    # Filter by multi mean
    catalog, skipped_records_multi = filter_multi_mean(
        catalog, multi_max, bypass_records
    )

    # Filter by fmax
    catalog, skipped_records_fmax = filter_fmax(catalog, fmax_min, bypass_records)

    # Filter by fmin
    catalog, skipped_records_fmin = filter_fmin(catalog, fmin_max, bypass_records)

    # Filter by missing station information
    catalog, skipped_records_sta = filter_missing_sta_info(catalog, bypass_records)

    # Filter by ground level locations
    catalog, skipped_records_ground = filter_ground_level_locations(
        catalog, bypass_records
    )

    # Filter by clipped records
    catalog, skipped_records_clipped = apply_clipNet_filter(
        catalog, clipped_records_ffp, bypass_records
    )

    # Filter by duplicate channels
    catalog, skipped_records_duplicate = filter_duplicate_channels(
        catalog, bypass_records
    )

    # Combine all the skipped records
    skipped_records = pd.concat(
        [
            skipped_records_has_score,
            skipped_records_score,
            skipped_records_multi,
            skipped_records_fmax,
            skipped_records_fmin,
            skipped_records_ground,
            skipped_records_duplicate,
        ]
    )

    return catalog, skipped_records


def create_quality_db(
    main_dir: Path,
    bypass_records_ffp: Path = None,
):
    """
    Create the quality database by running the following checks:
    1) Check there are GMC predictions
    2) Check against GMC predictions score mean
    3) Check against GMC predictions multi mean
    3) Check against GMC predictions fmax
    5) Check against GMC predictions fmin
    6) Ensure we use ground level locations
    7) Filter out clipped records
    8) Select which channel to use for duplicate HN, BN for the same evid / sta

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    bypass_records_ffp : Path, optional
        The file path to the records that will bypass the quality checks
    """
    # Make the quality db directory
    output_dir = main_dir / "quality_db"
    output_dir.mkdir(exist_ok=True)

    # Load the ground motion im catalog
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    gm_df = pd.read_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD50_FLAT,
        dtype={"evid": str},
    )

    # Get the clipped records
    clipped_records_ffp = (
        flatfile_dir / file_structure.SkippedRecordFilenames.CLIPPED_RECORDS
    )

    # Load the bypass records if they exist
    bypass_records = (
        pd.read_csv(bypass_records_ffp)["record_id"].to_numpy()
        if bypass_records_ffp
        else None
    )

    # Apply all the filters
    gm_df, skipped_records = apply_all_filters(
        gm_df, clipped_records_ffp, bypass_records
    )

    # Filter the other flatfiles based on the records in the rotd50_flat dataframe
    filter_flatfiles_on_catalouge(flatfile_dir, output_dir, gm_df)

    # Save the gm_df and skipped_records
    gm_df.to_csv(output_dir / FlatfileNames.GROUND_MOTION_IM_ROTD50_FLAT, index=False)
    skipped_records.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.QUALITY_SKIPPED_RECORDS,
        index=False,
    )
