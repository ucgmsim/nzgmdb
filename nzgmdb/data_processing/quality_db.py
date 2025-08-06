"""
Module to create the quality database for the NZGMDB.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from nzgmdb.management.data_registry import NZGMDB_DATA
from nzgmdb.management.file_structure import FlatfileNames
from oq_wrapper import constants, wrapper


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
        FlatfileNames.EARTHQUAKE_SOURCE_GEOMETRY,
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
        if file in [
            FlatfileNames.EARTHQUAKE_SOURCE_TABLE,
            FlatfileNames.EARTHQUAKE_SOURCE_GEOMETRY,
        ]:
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


def filter_has_score_mean(catalogue: pd.DataFrame, bypass_records: np.ndarray = None):
    """
    Filter the catalogue based on if there is a score from GMC.

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Find records that do not have a score value (is same across all components)
    has_score_filter = catalogue[catalogue["score_X"].isna()]

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

    return skipped_records


def filter_score_mean(
    catalogue: pd.DataFrame,
    score_min: float,
    bypass_records: np.ndarray = None,
    include_z: bool = False,
):
    """
    Filter the catalogue based on the score_mean value from GMC.
    Only looks at X and Y components by default, can include Z.

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    score_min : float
        The minimum score value to filter on
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks
    include_z : bool, optional
        Whether to include the Z component, by default False

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Find records that have too low of a score_X or score_Y value (or score_Z if include_z)
    score_min_filter = catalogue[
        (
            (catalogue["score_X"] < score_min)
            | (catalogue["score_Y"] < score_min)
            | (catalogue["score_Z"] < score_min)
            if include_z
            else (catalogue["score_X"] < score_min) | (catalogue["score_Y"] < score_min)
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

    return skipped_records


def filter_multi_mean(
    catalogue: pd.DataFrame,
    multi_max: float,
    bypass_records: np.ndarray = None,
    include_z: bool = False,
):
    """
    Filter the catalogue based on the multi_mean value
    Only looks at X and Y components by default, can include Z.

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    multi_max : float
        The maximum multi_mean value to filter on
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks
    include_z : bool, optional
        Whether to include the Z component, by default False

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Find records that have too high of a multi_X or multi_Y or multi_Z value
    multi_max_filter = catalogue[
        (
            (catalogue["multi_X"] > multi_max)
            | (catalogue["multi_Y"] > multi_max)
            | (catalogue["multi_Z"] > multi_max)
            if include_z
            else (
                (catalogue["multi_X"] > multi_max) | (catalogue["multi_Y"] > multi_max)
            )
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

    return skipped_records


def filter_fmax(
    catalogue: pd.DataFrame, fmax_min: float, bypass_records: np.ndarray = None
):
    """
    Filter the catalogue based on the fmax_min value for the fmax_X and fmax_Y.
    (Horizontal components only, vertical component is not considered)

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    fmax_min : float
        The minimum fmax value to filter on horizontal components
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Find fmax_min
    catalogue.loc[:, "fmax_min"] = catalogue[["fmax_X", "fmax_Y"]].apply(min, axis=1)

    # Find records that have too low of a fmax_min value
    fmax_min_filter = catalogue[catalogue["fmax_min"] < fmax_min]

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

    return skipped_records


def filter_fmin(
    catalogue: pd.DataFrame, fmin_max: float, bypass_records: np.ndarray = None
):
    """
    Filter the catalogue based on the fmin max value for the fmin_X and fmin_Y.
    (Horizontal components only, vertical component is not considered)

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    fmin_max : float
        The maximum fmin value to filter on horizontal components
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Find records that have too high of a fmin_max value
    fmin_max_filter = catalogue[catalogue[["fmin_X", "fmin_Y"]].max(axis=1) > fmin_max]

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

    return skipped_records


def filter_missing_sta_info(
    catalogue: pd.DataFrame, bypass_records: np.ndarray | None = None
):
    """
    Filter the catalogue based on the missing station information

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Find records that are missing station information
    missing_sta_filter = catalogue[catalogue["Vs30"].isna()]

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

    return skipped_records


def filter_ground_level_locations(
    catalogue: pd.DataFrame, bypass_records: np.ndarray = None
):
    """
    Filter the catalogue based on the ground level locations

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks


    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Filter records that are not ground level
    ground_level_filter = catalogue[~catalogue["is_ground_level"]]

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

    return skipped_records


def apply_clipNet_filter(
    catalogue: pd.DataFrame,
    clipped_records_ffp: Path,
    bypass_records: np.ndarray = None,
):
    """
    Apply the ClipNet filter to the catalogue
    Removes the clipped records from the catalogue and creates a skipped records dataframe

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    clipped_records_ffp : Path
        The file path to the clipped records (created during the GeoNet processing)
    bypass_records : np.ndarray, optional
        The records to bypass the quality

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Read the clipped records
    try:
        clipped_records = pd.read_csv(clipped_records_ffp)
    except pd.errors.EmptyDataError:
        return catalogue, pd.DataFrame(columns=["record_id", "reason"])

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

    return skipped_records


def filter_troublesome_sensitivity(
    catalogue: pd.DataFrame, bypass_records: np.ndarray = None
):
    """
    Filter the catalogue by removing records that are known to be troublesome for sensitivity analysis.

    This function removes records that have been identified as problematic for sensitivity analysis,
    such as those with incorrect values assigned during first deployment of broadband instruments.

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Load the sensitivity ignore file from the data registry
    sensitivity_ignore = pd.read_csv(NZGMDB_DATA.fetch("sensitivity_ignore.csv"))

    # Ensure datetime columns are in datetime format
    catalogue["datetime"] = pd.to_datetime(catalogue["datetime"])
    sensitivity_ignore["start_date"] = pd.to_datetime(sensitivity_ignore["start_date"])
    sensitivity_ignore["end_date"] = pd.to_datetime(sensitivity_ignore["end_date"])

    # Ensure the dtypes are correct for merging
    for col in ["sta", "chan", "loc"]:
        catalogue[col] = catalogue[col].astype(str)
        sensitivity_ignore[col] = sensitivity_ignore[col].astype(str)

    # Merge on sta, chan, loc to find records that have the same sta chan and loc
    # as the sensitivity ignore records
    merged = pd.merge(
        catalogue,
        sensitivity_ignore,
        on=["sta", "chan", "loc"],
        how="inner",
        suffixes=("", "_ignore"),
    )

    # Filter where catalogue datetime is within the ignore period
    sensitivity_filter = merged[
        (merged["datetime"] >= merged["start_date"])
        & (merged["datetime"] <= merged["end_date"])
    ]

    # Remove the bypass records if they exist
    if bypass_records is not None:
        sensitivity_filter = sensitivity_filter[
            ~sensitivity_filter["record_id"].isin(bypass_records)
        ]

    # Create the skipped_records dataframe from sensitivity_filter
    skipped_records = pd.DataFrame(
        {
            "record_id": sensitivity_filter["record_id"],
            "reason": "Troublesome sensitivity record",
        }
    )

    return skipped_records


def filter_empirical_predictions(
    catalogue: pd.DataFrame,
    bypass_records: np.ndarray = None,
    mean_residual_threshold: float = None,
    max_residual_threshold: float = None,
):
    """
    This function checks the difference in empirical estimated values and the results from
    the catalogue. If the difference exceeds certain thresholds, the record is skipped.
    Note: the empirical predictions are based on the Atkinson 2022 model for subduction interface and slab,
    and crustal for any other tectonic type.

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks
    mean_residual_threshold : float, optional
        The threshold for the mean residual difference, by default grabs from the config
    max_residual_threshold : float, optional
        The threshold for the max residual difference, by default grabs from the config

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Extract the periods from the catalogue 0.01 -> 10.0
    psa_cols = [col for col in catalogue.columns if col.startswith("pSA")]
    # Extract numeric values and filter between 0.01 and 10.0
    psa_periods = [
        float(col.split("_")[1])
        for col in psa_cols
        if 0.01 <= float(col.split("_")[1]) <= 10.0
    ]
    # Get the filtered column names for comparison
    psa_cols_filtered = [f"pSA_{num}" for num in psa_periods]

    # Split the catalogue into different tectonic types
    interface_catalogue = catalogue[catalogue["tect_class"] == "Interface"]
    slab_catalogue = catalogue[catalogue["tect_class"] == "Slab"]
    # Classify any other tectonic types as crustal
    crustal_catalogue = catalogue[~catalogue["tect_class"].isin(["Interface", "Slab"])]

    # For each tectonic type, run the empirical predictions
    to_run = {
        constants.TectType.SUBDUCTION_INTERFACE: interface_catalogue,
        constants.TectType.SUBDUCTION_SLAB: slab_catalogue,
        constants.TectType.ACTIVE_SHALLOW: crustal_catalogue,
    }

    im_emp = []

    for tect_type, tect_catalogue in to_run.items():
        # Grab the Empirical predictions from the catalogue, based on Atkinson 2022
        input_df = pd.DataFrame(
            {
                "mag": tect_catalogue["mag"],
                "rrup": tect_catalogue["r_rup"],
                "vs30": tect_catalogue["Vs30"],
                "z1pt0": tect_catalogue["Z1.0"] / 1000,  # Convert to km
                "backarc": [False] * len(tect_catalogue),
            }
        )
        result_df = wrapper.run_gmm(
            constants.GMM.A_22,
            tect_type,
            input_df,
            "pSA",
            periods=psa_periods,
        )

        # Extract all results that have _mean at the end of the column name and rename them to remove the _mean
        im_emp_tect_type = result_df.filter(like="_mean").rename(
            columns=lambda x: x.replace("_mean", "")
        )

        # Add the record_id to the empirical predictions
        im_emp_tect_type["record_id"] = tect_catalogue["record_id"].values

        # Append the empirical predictions to the list
        im_emp.append(im_emp_tect_type)

    # Concatenate the empirical predictions for all tectonic types
    im_emp = pd.concat(im_emp, ignore_index=True)

    # Order by record_id in the im_emp to match the order of catalogue
    im_emp = im_emp.set_index("record_id").loc[catalogue["record_id"]].reset_index()

    # Compute the log-difference
    # Note: im_emp is already in logspace, so no need to convert it
    residual_diff = (
        np.log(catalogue.loc[:, psa_cols_filtered]) - im_emp.loc[:, psa_cols_filtered]
    )

    # Calculate mean and max residuals using the precomputed difference
    catalogue["mean_residual"] = np.abs(residual_diff.mean(axis=1))
    catalogue["max_residual"] = np.abs(
        residual_diff.max(axis=1) - residual_diff.min(axis=1)
    )

    # Obtain the thresholds from parameters or use defaults from the config
    config = cfg.Config()
    mean_residual_threshold = (
        config.get_value("mean_residual_threshold")
        if mean_residual_threshold is None
        else mean_residual_threshold
    )
    max_residual_threshold = (
        config.get_value("max_residual_threshold")
        if max_residual_threshold is None
        else max_residual_threshold
    )

    # Create filters based on the thresholds
    filters = {
        "max_residual": {
            "threshold": max_residual_threshold,
            "reason": f"Empirical predictions max_residual exceeds threshold {max_residual_threshold}",
        },
        "mean_residual": {
            "threshold": mean_residual_threshold,
            "reason": f"Empirical predictions mean_residual exceeds threshold {mean_residual_threshold}",
        },
    }

    # Apply the filters to the catalogue
    skipped_list = []
    for col, params in filters.items():
        filt = catalogue[catalogue[col] > params["threshold"]]
        if bypass_records is not None:
            filt = filt[~filt["record_id"].isin(bypass_records)]
        skipped = pd.DataFrame(
            {
                "record_id": filt["record_id"],
                "reason": params["reason"],
            }
        )
        skipped_list.append(skipped)

    # Concatenate all skipped records
    skipped_records = pd.concat(skipped_list, ignore_index=True)

    return skipped_records


def filter_duplicate_channels(
    catalogue: pd.DataFrame, bypass_records: np.ndarray = None
):
    """
    Filter the catalogue by removing lower-priority duplicate channel records.

    For each (evid, sta) combination that has multiple channel entries, the function
    keeps only one record based on the following priority:
    1. Records listed in `bypass_records` (highest priority)
    2. HN channels (Strong motion, high frequency)
    3. BN channels (Strong motion, lower frequency)
    4. HH channels (Broadband, high frequency)

    If multiple records have the same priority, the first one encountered is kept.
    All other duplicates are removed and returned in the skipped records.

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks

    Returns
    -------
    pd.DataFrame
        The skipped records to filter out of the catalogue
    """
    # Step 1: Create 'evid_sta' for grouping
    catalogue["evid_sta"] = catalogue["evid"].astype(str) + "_" + catalogue["sta"]

    # Step 2: Create bypass flag using record_id
    if bypass_records is None:
        bypass_records = []
    catalogue["bypass"] = catalogue["record_id"].isin(bypass_records)

    # Step 3: Define priority levels
    priority = {"HN": 1, "BN": 2, "HH": 3}
    catalogue["chan_priority"] = catalogue["chan"].map(priority).fillna(4)
    # Step 4: Override priority for bypass records
    catalogue.loc[catalogue["bypass"], "chan_priority"] = 0

    # Step 5: Sort by priority and select top-priority row per group
    catalog_sorted = catalogue.sort_values(by=["evid_sta", "chan_priority"])
    # Remove records with priority 4 (not HN, BN, HH)
    catalog_sorted = catalog_sorted[catalog_sorted["chan_priority"] < 4]
    best_dups = catalog_sorted.groupby("evid_sta", as_index=False).nth(0)

    # Step 6: Identify which records to drop (the non-best ones)
    records_to_keep = best_dups["record_id"]
    records_to_drop = catalogue[~catalogue["record_id"].isin(records_to_keep)]

    # Step 7: Prepare skipped_records
    skipped_records = pd.DataFrame(
        {"record_id": records_to_drop["record_id"], "reason": "Duplicate channels"}
    )

    return skipped_records


def apply_all_filters(
    catalogue: pd.DataFrame,
    clipped_records_ffp: Path,
    bypass_records: np.ndarray = None,
    score_min: float = None,
    multi_max: float = None,
    fmax_min: float = None,
    fmin_max: float = None,
):
    """
    Apply all the quality filters to the catalogue.

    This function performs the following filtering steps:
    1) Ensure only ground level locations are used.
    2) Filter by presence of GMC predictions.
    3) Filter by score mean.
    4) Filter by multi mean.
    5) Filter by fmax.
    6) Filter by fmin.
    7) Filter by missing station information.
    8) Filter out clipped records.
    9) Filter out troublesome sensitivity records.
    10) Filter out records too far from empirical predictions.
    11) Select the appropriate channel for duplicate HN/BN records for the same event/station.

    Parameters
    ----------
    catalogue : pd.DataFrame
        The catalogue dataframe to filter.
    clipped_records_ffp : Path
        The file path to the clipped records (created during GeoNet processing).
    bypass_records : np.ndarray, optional
        The records to bypass the quality checks.
    score_min : float, optional
        The minimum score value to filter on.
    multi_max : float, optional
        The maximum multi_mean value to filter on.
    fmax_min : float, optional
        The minimum fmax value to filter on.
    fmin_max : float, optional
        The maximum fmin value to filter on.

    Returns
    -------
    pd.DataFrame
        The filtered catalogue.
    pd.DataFrame
        The skipped records to filter out of the catalogue.
    """

    config = cfg.Config()

    # Get the config values if they are not provided
    score_min = score_min if score_min is not None else config.get_value("score_min")
    multi_max = multi_max if multi_max is not None else config.get_value("multi_max")
    fmax_min = fmax_min if fmax_min is not None else config.get_value("fmax_min")
    fmin_max = fmin_max if fmin_max is not None else config.get_value("fmin_max")

    # Find ground level locations
    skipped_records_ground = filter_ground_level_locations(catalogue, bypass_records)

    # Find has score mean
    skipped_records_has_score = filter_has_score_mean(catalogue, bypass_records)

    # Find score mean
    skipped_records_score = filter_score_mean(catalogue, score_min, bypass_records)

    # Find multi mean
    skipped_records_multi = filter_multi_mean(catalogue, multi_max, bypass_records)

    # Find fmax
    skipped_records_fmax = filter_fmax(catalogue, fmax_min, bypass_records)

    # Find fmin
    skipped_records_fmin = filter_fmin(catalogue, fmin_max, bypass_records)

    # Find missing station information
    skipped_records_sta = filter_missing_sta_info(catalogue, bypass_records)

    # Find clipped records
    skipped_records_clipped = apply_clipNet_filter(
        catalogue, clipped_records_ffp, bypass_records
    )

    # Find troublesome sensitivity records
    skipped_records_sensitivity = filter_troublesome_sensitivity(
        catalogue, bypass_records
    )

    # Find empirical predictions
    skipped_records_empirical = filter_empirical_predictions(catalogue, bypass_records)

    # Combine all the skipped records
    skipped_records = pd.concat(
        [
            skipped_records_ground,
            skipped_records_has_score,
            skipped_records_score,
            skipped_records_multi,
            skipped_records_fmax,
            skipped_records_fmin,
            skipped_records_sta,
            skipped_records_clipped,
            skipped_records_sensitivity,
            skipped_records_empirical,
        ]
    )

    # Filter out the skipped records from the catalogue
    catalogue = catalogue[~catalogue["record_id"].isin(skipped_records["record_id"])]

    # Find duplicate channels
    skipped_records_duplicate = filter_duplicate_channels(catalogue, bypass_records)

    # Filter out the duplicate channels from the catalogue
    catalogue = catalogue[
        ~catalogue["record_id"].isin(skipped_records_duplicate["record_id"])
    ]

    # Add the skipped records from duplicate channels
    skipped_records = pd.concat([skipped_records, skipped_records_duplicate])

    # Clean up and ensure uniqueness
    assert len(catalogue["evid_sta"].unique()) == len(catalogue)
    catalogue = catalogue.drop(
        columns=[
            "evid_sta",
            "bypass",
            "chan_priority",
            "mean_residual",
            "max_residual",
            "fmax_min",
        ]
    )

    return catalogue, skipped_records


def create_quality_db(
    main_dir: Path,
    bypass_records_ffp: Path = None,
):
    """
    Create the quality database by running the following checks:
    1) Filter by presence of GMC predictions.
    2) Filter by score mean.
    3) Filter by multi mean.
    4) Filter by fmax.
    5) Filter by fmin.
    6) Filter by missing station information.
    7) Ensure only ground level locations are used.
    8) Filter out clipped records.
    9) Filter out troublesome sensitivity records.
    10) Filter out records too far from empirical predictions.
    11) Select the appropriate channel for duplicate HN/BN records for the same event/station.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    bypass_records_ffp : Path, optional
        The file path to the records that will bypass the quality checks
    """
    # Make the quality db directory
    output_dir = main_dir / "quality_db_testing"
    output_dir.mkdir(exist_ok=True)

    # Load the ground motion im catalogue
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    gm_df = pd.read_csv(
        flatfile_dir / file_structure.FlatfileNames.GROUND_MOTION_IM_ROTD50_FLAT,
        dtype={"evid": str},
    )

    # Get the clipped records
    # clipped_records_ffp = (
    #     flatfile_dir / file_structure.SkippedRecordFilenames.CLIPPED_RECORDS
    # )
    clipped_records_ffp = Path("/home/joel/local/gmdb/4p3_mantle/tmp_clipped.csv")

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
