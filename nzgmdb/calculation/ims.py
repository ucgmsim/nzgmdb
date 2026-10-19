"""
This module contains functions for calculating the Intensity Measures (IMs) for the NZGMDB records.
"""

import functools
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from IM import im_calculation, ims, waveform_reading
from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure

# Components saved for each IM table
FAS_COMPONENTS = ["000", "090", "ver", "geom", "eas"]
PSA_COMPONENTS = ["000", "090", "ver", "geom"]
ROTD_COMPONENTS = ["rotd0", "rotd50", "rotd100"]
# Rotation angle (degrees) of each RotD0 / RotD100 IM, saved as their own components
ORIENTATION_COMPONENTS = ["rotd0_orientation", "rotd100_orientation"]

# IMs that are not saved for the RotD components
ROTD_EXCLUDED_IMS = ["CAV", "CAV5", "AI", "Ds575", "Ds595"]

# Columns describing the record at the front of every IM table
RECORD_COLUMNS = ["record_id", "component", "evid", "sta", "chan", "loc"]


def calculate_im_for_record(
    ffp_000: Path,
    intensity_measures: list[ims.IM],
    psa_periods: np.ndarray,
    fas_frequencies: np.ndarray,
    bandwidth: float,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Calculate the IMs for a single record.

    Parameters
    ----------
    ffp_000 : Path
        The full file path to the 000 component file
    intensity_measures : list[ims.IM]
        The list of intensity measures to calculate
    psa_periods : np.ndarray
        The periods for calculating the pseudo-spectral acceleration
    fas_frequencies : np.ndarray
        The frequencies for calculating the Fourier amplitude spectrum
    bandwidth : float
        The Konno-Ohmachi smoothing bandwidth for the Fourier amplitude spectrum

    Returns
    -------
    im_df : pd.DataFrame | None
        The IMs with a row per component and record_id / component columns at the front
        None if the record was skipped
    skipped_record : pd.DataFrame | None
        The record_id and reason the record was skipped, None if it was not skipped
    """
    # Get the 090 and ver components full file paths
    ffp_090 = ffp_000.parent / f"{ffp_000.stem}.090"
    ffp_ver = ffp_000.parent / f"{ffp_000.stem}.ver"
    record_id = ffp_000.stem

    try:
        dt, waveform = waveform_reading.read_ascii(ffp_000, ffp_090, ffp_ver)
    except FileNotFoundError:
        skipped_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "reason": "Failed to find the waveform ascii files",
                }
            ]
        )
        return None, skipped_record
    except ValueError:
        skipped_record = pd.DataFrame(
            [{"record_id": record_id, "reason": "Components contain NaN values"}]
        )
        return None, skipped_record

    # Calculate the IMs (FAS frequencies above Nyquist are dropped by IM_calculation with a warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        im_df = im_calculation.calculate_ims(
            waveform,
            dt,
            intensity_measures,
            psa_periods,
            fas_frequencies,
            bandwidth=bandwidth,
        )

    # Set the component and record_id columns at the front
    im_df = im_df.rename_axis("component").reset_index()
    im_df.insert(0, "record_id", record_id)

    return im_df, None


def split_component_tables(
    im_df: pd.DataFrame, fas_columns: list[str]
) -> dict[str, pd.DataFrame]:
    """
    Split the IM results into the tables saved for each component.

    Parameters
    ----------
    im_df : pd.DataFrame
        The IM results for a set of records, with record_id and component columns
    fas_columns : list[str]
        The FAS columns to save, so every batch has the same columns
        Empty if FAS was not calculated, then no FAS tables are made

    Returns
    -------
    dict[str, pd.DataFrame]
        The table for each component, keyed by the component name used in
        file_structure.get_im_component_dir, e.g. "rotd50" or "000_fas"
    """
    # Split the record id into evid, sta, chan, loc
    record_parts = im_df["record_id"].str.split("_", n=3, expand=True)
    record_parts.columns = ["evid", "sta", "chan", "loc"]
    im_df = pd.concat([im_df, record_parts], axis=1)

    im_columns = [
        col
        for col in im_df.columns
        if col not in RECORD_COLUMNS and not col.startswith("FAS_")
    ]
    rotd_columns = [col for col in im_columns if col not in ROTD_EXCLUDED_IMS]

    # Each table type with the components and IM columns it keeps
    table_types = [
        (PSA_COMPONENTS, "", im_columns),
        (ROTD_COMPONENTS + ORIENTATION_COMPONENTS, "", rotd_columns),
    ]
    if fas_columns:
        table_types.append((FAS_COMPONENTS, "_fas", fas_columns))

    tables = {}
    for components, suffix, columns in table_types:
        for component in components:
            component_rows = im_df[im_df["component"] == component]
            if not component_rows.empty:
                tables[f"{component}{suffix}"] = component_rows.reindex(
                    columns=RECORD_COLUMNS + columns
                ).reset_index(drop=True)

    return tables


def get_skipped_records_dir(main_dir: Path) -> Path:
    """
    Get the directory of the skipped records file for each IM batch.

    A batch's skipped records file is written last, so it marks the batch as complete.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results

    Returns
    -------
    Path
        The directory of batch_<n>.csv skipped records files
    """
    return file_structure.get_im_merge_dir(main_dir) / "im_calc_skipped_records"


def prepare_batches(main_dir: Path, batch_records: pd.DataFrame):
    """
    Save the record_id / batch assignment for this run. If it differs from the saved
    assignment (e.g. a different batch size or set of records), the batch files of the
    earlier run are removed so they are not resumed or mixed in, as the component
    directories are read as a whole by later steps.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results
    batch_records : pd.DataFrame
        The record_id and batch of every record in this run
    """
    im_merge_dir = file_structure.get_im_merge_dir(main_dir)
    batch_records_ffp = im_merge_dir / "batch_records.csv"

    if batch_records_ffp.exists():
        saved_batch_records = pd.read_csv(batch_records_ffp, dtype={"record_id": str})
        if saved_batch_records["record_id"].tolist() == batch_records[
            "record_id"
        ].tolist() and (
            saved_batch_records["batch"].tolist() == batch_records["batch"].tolist()
        ):
            return
        print("IM batches differ from the earlier run, removing its batch files")
        for batch_file in im_merge_dir.glob("*/batch_*"):
            batch_file.unlink()

    im_merge_dir.mkdir(parents=True, exist_ok=True)
    batch_records.to_csv(batch_records_ffp, index=False)


def compute_ims_for_all_processed_records(
    main_dir: Path,
    n_procs: int = 1,
    intensity_measures: list[ims.IM] | None = None,
    batch_size: int = 5000,
):
    """
    Compute the IMs for all processed records in batches and save a parquet file per batch
    for each component, which later steps read as one dataset per component directory.

    Batches that already have a skipped records file are skipped (checkpointing),
    as long as the records in each batch are the same as the earlier run.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    n_procs : int, optional
        The number of processes to use
    intensity_measures : list[ims.IM], optional
        The list of intensity measures to calculate, by default None and will use the config file
    batch_size : int, optional
        The number of records in each batch, by default 5000
    """
    # Get the waveform directory and all the 000 files (sorted so batches are the same on a re-run)
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    comp_000_files = sorted(waveform_dir.rglob("*.000"))

    # At least 1 batch so a run with no records still writes the skipped records file
    n_batches = max(1, int(np.ceil(len(comp_000_files) / batch_size)))
    batches = np.array_split(np.asarray(comp_000_files, dtype=object), n_batches)
    print(f"Calculating IMs for {len(comp_000_files)} records in {n_batches} batches")

    # Load the config and extract the IM options
    config = cfg.Config()
    intensity_measures = (
        [ims.IM[measure] for measure in config.get_value("ims")]
        if intensity_measures is None
        else intensity_measures
    )
    psa_periods = np.asarray(config.get_value("psa_periods"))
    fas_frequencies = np.logspace(
        np.log10(config.get_value("common_frequency_start")),
        np.log10(config.get_value("common_frequency_end")),
        num=config.get_value("common_frequency_num"),
    )
    fas_columns = (
        [f"FAS_{im_calculation.frequency_label(freq)}" for freq in fas_frequencies]
        if ims.IM.FAS in intensity_measures
        else []
    )

    record_fn = functools.partial(
        calculate_im_for_record,
        intensity_measures=intensity_measures,
        psa_periods=psa_periods,
        fas_frequencies=fas_frequencies,
        bandwidth=config.get_value("ko_bandwidth"),
    )

    prepare_batches(
        main_dir,
        pd.DataFrame(
            {
                "record_id": [ffp_000.stem for batch in batches for ffp_000 in batch],
                "batch": [
                    batch_index
                    for batch_index, batch in enumerate(batches)
                    for _ in batch
                ],
            }
        ),
    )
    skipped_records_dir = get_skipped_records_dir(main_dir)
    skipped_records_dir.mkdir(parents=True, exist_ok=True)

    for batch_index, batch in enumerate(batches):
        batch_skipped_ffp = skipped_records_dir / f"batch_{batch_index}.csv"
        if batch_skipped_ffp.exists():
            print(f"Skipping IM batch {batch_index + 1}/{n_batches} (already complete)")
            continue

        print(f"Calculating IMs for batch {batch_index + 1}/{n_batches}")
        if n_procs == 1:
            results = [record_fn(ffp_000) for ffp_000 in batch]
        else:
            # forkserver, as forking this multi-threaded process can corrupt the worker heap
            with mp.get_context("forkserver").Pool(n_procs) as p:
                results = p.map(record_fn, batch)

        im_dfs = [im_df for im_df, _ in results if im_df is not None]
        skipped_dfs = [skipped for _, skipped in results if skipped is not None]

        # Save a parquet file for this batch in each component directory
        if im_dfs:
            tables = split_component_tables(
                pd.concat(im_dfs, ignore_index=True), fas_columns
            )
            for component, table in tables.items():
                component_dir = file_structure.get_im_component_dir(main_dir, component)
                component_dir.mkdir(parents=True, exist_ok=True)
                table.to_parquet(
                    component_dir / f"batch_{batch_index}.parquet",
                    compression="gzip",
                    index=False,
                )

        # Save the skipped records last, as it marks the batch as complete
        skipped_df = (
            pd.concat(skipped_dfs, ignore_index=True)
            if skipped_dfs
            else pd.DataFrame(columns=["record_id", "reason"])
        )
        skipped_df.to_csv(batch_skipped_ffp, index=False)

    print("Finished calculating IMs")

    # Combine the skipped records of this run's batches
    skipped_records_df = pd.concat(
        [
            pd.read_csv(skipped_records_dir / f"batch_{batch_index}.csv")
            for batch_index in range(n_batches)
        ],
        ignore_index=True,
    )
    skipped_records_df.to_csv(
        file_structure.get_flatfile_dir(main_dir)
        / file_structure.SkippedRecordFilenames.IM_CALC_SKIPPED_RECORDS,
        index=False,
    )
