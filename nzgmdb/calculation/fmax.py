"""
Calculates the maximum useable frequency (fmax).
"""

import functools
import multiprocessing as mp
from enum import StrEnum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors, file_structure
from nzgmdb.mseed_management import reading

FMAX_COLUMNS = [
    "record_id",
    "fmax_000",
    "fmax_090",
    "fmax_ver",
    "fmax_status",
    "lpf_default",
]


class FmaxStatus(StrEnum):
    """How the fmax of a record was found, saved in the fmax_status column."""

    COMPUTED = "computed"
    SCREENING_FAILED = "screening_failed"
    NO_SNR = "no_snr"
    INVALID_MSEED = "invalid_mseed"
    BYPASS = "bypass"


def nan_fmax_record(
    record_id: str, status: FmaxStatus, lpf_default: float = np.nan
) -> pd.DataFrame:
    """
    Fmax row for a record whose fmax could not be computed.

    Parameters
    ----------
    record_id : str
        The record id
    status : FmaxStatus
        Why the fmax could not be computed
    lpf_default : float, optional
        The default lowpass filter frequency (scaled Nyquist) used in processing
        when there is no fmax, by default NaN

    Returns
    -------
    pd.DataFrame
        Single row with NaN fmax values and the status
    """
    return pd.DataFrame(
        [
            {
                "record_id": record_id,
                "fmax_000": np.nan,
                "fmax_090": np.nan,
                "fmax_ver": np.nan,
                "fmax_status": status.value,
                "lpf_default": lpf_default,
            }
        ]
    )


def brune_f0(mag: float) -> float:
    """
    Brune corner frequency of an event, clipped to the config f0_min_Hz - f0_max_Hz.

    Parameters
    ----------
    mag : float
        Event magnitude, used as the moment magnitude

    Returns
    -------
    float
        Corner frequency in Hz
    """
    config = cfg.Config()
    m0 = 10 ** (1.5 * mag + 16.05)
    f0 = (
        4.906e6
        * config.get_value("f0_beta_kms")
        * (config.get_value("f0_stress_drop_bar") / m0) ** (1 / 3)
    )
    return float(
        np.clip(f0, config.get_value("f0_min_Hz"), config.get_value("f0_max_Hz"))
    )


def get_search_start_freqs(
    meta_output_dir: Path, mseed_files: list[Path]
) -> list[float | None]:
    """
    Get the frequency each record's fmax search starts from.

    Parameters
    ----------
    meta_output_dir : Path
        Path to the flatfile directory containing the tectonic earthquake source table
    mseed_files : list[Path]
        The mseed files to process

    Returns
    -------
    list[float | None]
        The start frequency for each file, None uses the config min_freq_Hz
        (fmax_search_start "fixed", or no magnitude for the event)
    """
    config = cfg.Config()
    if config.get_value("fmax_search_start") != "f0":
        return [None] * len(mseed_files)

    event_df = pd.read_csv(
        meta_output_dir
        / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_TECTONIC,
        dtype={"evid": str},
    )
    f0_per_event = {
        evid: brune_f0(mag)
        for evid, mag in zip(event_df["evid"], event_df["mag"])
        if not pd.isna(mag)
    }
    return [f0_per_event.get(ffp.stem.split("_")[0]) for ffp in mseed_files]


def run_full_fmax_calc(
    meta_output_dir: Path,
    waveform_dir: Path,
    snr_fas_output_dir: Path,
    batches_dir: Path,
    n_procs: int = 1,
    bypass_records_ffp: Path | None = None,
    batch_size: int = 20000,
):
    """
    Run the full procedure for each record to assess SNR produced from mseed files
    and get the maximum usable frequency (fmax).

    Parameters
    ----------
    meta_output_dir : Path
        Path to the output directory for the metadata and skipped records.
    waveform_dir : Path
        Path to the directory containing the mseed files to process.
    snr_fas_output_dir : Path
        Path to the output directory for the SNR and FAS data.
    batches_dir : Path
        Path to the directory where batch results will be stored for checkpointing.
    n_procs : int, optional
        Number of processes to use, by default 1.
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file, which includes a custom fmax
    batch_size : int, optional
        The number of mseed files to process in each batch, by default 20000.
    """
    batches_dir.mkdir(parents=True, exist_ok=True)

    # Sorted so the batches are the same on a re-run
    mseed_files = sorted(waveform_dir.rglob("*.mseed"))
    search_start_freqs = get_search_start_freqs(meta_output_dir, mseed_files)
    total = len(mseed_files)
    print(f"Total number of mseed files to process: {total}")

    # Create batches of mseed files for checkpointing (at least 1 so no files still writes empty outputs)
    n_batches = max(1, int(np.ceil(total / batch_size)))
    mseed_batches = np.array_split(np.asarray(mseed_files, dtype=object), n_batches)
    start_freq_batches = np.array_split(
        np.asarray(search_start_freqs, dtype=object), n_batches
    )

    for index, (batch, start_freq_batch) in enumerate(
        zip(mseed_batches, start_freq_batches)
    ):
        batch_dir = batches_dir / f"batch_{index + 1}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_fmax_file = batch_dir / file_structure.FlatfileNames.FMAX
        if not batch_fmax_file.exists():
            print(f"Processing batch {index + 1}/{len(mseed_batches)}")
            with mp.get_context("forkserver").Pool(n_procs) as p:
                results = p.starmap(
                    functools.partial(
                        assess_snr_and_get_fmax,
                        snr_fas_output_dir=snr_fas_output_dir,
                    ),
                    zip(batch, start_freq_batch),
                )

            if len(results) == 0:
                print(f"Batch {index + 1}: No records to process")
                meta_dfs, skipped_record_dfs = [None], [None]
            else:
                # Unpack the results
                meta_dfs, skipped_record_dfs = zip(*results)

            # Check that there are dataframes that are not None before concatenating
            if not all(value is None for value in meta_dfs):
                fmax_df = pd.concat(meta_dfs).reset_index(drop=True)
            else:
                fmax_df = pd.DataFrame(columns=FMAX_COLUMNS)
            if not all(value is None for value in skipped_record_dfs):
                skipped_records_df = pd.concat(skipped_record_dfs).reset_index(
                    drop=True
                )
            else:
                skipped_records_df = pd.DataFrame(columns=["record_id", "reason"])

            print(f"Batch {index + 1}: Skipped {len(skipped_records_df)} records")
            skipped_records_df.to_csv(
                batch_dir / file_structure.SkippedRecordFilenames.FMAX_SKIPPED_RECORDS,
                index=False,
            )

            # Check if the bypass records file is provided
            if bypass_records_ffp is not None:
                # Add the bypass records to the fmax_df, or overwrite the fmax values if the record is in the bypass records
                bypass_df = pd.read_csv(bypass_records_ffp)
                fmax_df = fmax_df.merge(
                    bypass_df[["record_id", "fmax_000", "fmax_090", "fmax_ver"]],
                    on="record_id",
                    how="left",
                    suffixes=("", "_bypass"),
                )
                has_bypass_fmax = (
                    fmax_df[["fmax_000_bypass", "fmax_090_bypass", "fmax_ver_bypass"]]
                    .notna()
                    .any(axis=1)
                )
                fmax_df.loc[has_bypass_fmax, "fmax_status"] = FmaxStatus.BYPASS.value
                fmax_df["fmax_000"] = fmax_df["fmax_000_bypass"].fillna(
                    fmax_df["fmax_000"]
                )
                fmax_df["fmax_090"] = fmax_df["fmax_090_bypass"].fillna(
                    fmax_df["fmax_090"]
                )
                fmax_df["fmax_ver"] = fmax_df["fmax_ver_bypass"].fillna(
                    fmax_df["fmax_ver"]
                )
                fmax_df = fmax_df.drop(
                    columns=[
                        "fmax_000_bypass",
                        "fmax_090_bypass",
                        "fmax_ver_bypass",
                    ]
                )

            fmax_df.to_csv(batch_dir / file_structure.FlatfileNames.FMAX, index=False)

    # Combine only this run's batches, so batches from a run with a different batch_size are not mixed in
    batch_dirs = [batches_dir / f"batch_{index + 1}" for index in range(n_batches)]
    all_skipped_records_df = pd.concat(
        [
            pd.read_csv(
                batch_dir / file_structure.SkippedRecordFilenames.FMAX_SKIPPED_RECORDS
            )
            for batch_dir in batch_dirs
        ],
        ignore_index=True,
    )
    all_fmax_df = pd.concat(
        [
            pd.read_csv(batch_dir / file_structure.FlatfileNames.FMAX)
            for batch_dir in batch_dirs
        ],
        ignore_index=True,
    )

    all_skipped_records_df.to_csv(
        meta_output_dir / file_structure.SkippedRecordFilenames.FMAX_SKIPPED_RECORDS,
        index=False,
    )
    all_fmax_df.to_csv(meta_output_dir / file_structure.FlatfileNames.FMAX, index=False)


def assess_snr_and_get_fmax(
    filename: Path,
    search_start_freq: float | None = None,
    snr_fas_output_dir: Path | None = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Assess the record's Signal-to-Noise Ratio (SNR) within the frequency interval specified in the config file
    and determine the record's maximum usable frequency (fmax) if its SNR exceeds the threshold.

    Parameters
    ----------
    filename : Path
        Path to the `record_id.mseed` file.
    search_start_freq : float, optional
        Frequency the fmax search starts from, by default None uses the config min_freq_Hz
    snr_fas_output_dir : Path
        Path to the output directory where the `record_id_snr_fas.csv` file is stored.

    Returns
    -------
    fmax_record: pd.DataFrame
        DataFrame containing the fmax values, fmax_status and lpf_default
        (the scaled Nyquist lowpass used in processing when there is no fmax).
        If the fmax could not be computed (e.g. screening failed) the fmax values are NaN.
    skipped_record: pd.DataFrame | None
        If the record is skipped, returns a DataFrame containing the reason for skipping, otherwise returns None.
    """
    config = cfg.Config()
    record_id = filename.stem

    # read the mseed file to get the delta
    try:
        mseed = reading.read_mseed_to_stream(filename)
    except custom_errors.InvalidMseedFileError:
        fmax_record = nan_fmax_record(record_id, FmaxStatus.INVALID_MSEED)
        skipped_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "reason": "Invalid mseed file",
                }
            ]
        )
        return fmax_record, skipped_record
    dt = mseed[0].stats.delta

    # current_row["delta"] is a pd.Series() containing 1 float so .iloc[0]
    # is used to get the float from the pd.Series()
    scaled_nyquist_freq = (
        (1 / dt) * 0.5 * config.get_value("nyquist_freq_scaling_factor")
    )

    snr_filename = file_structure.get_snr_fas_ffp(snr_fas_output_dir, filename)

    if not snr_filename.exists():
        fmax_record = nan_fmax_record(record_id, FmaxStatus.NO_SNR, scaled_nyquist_freq)
        skipped_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "reason": "No SNR file found for record",
                }
            ]
        )
        return fmax_record, skipped_record

    # Assess the SNR
    snr_with_freq_signal_noise = pd.read_csv(snr_filename)
    snr = snr_with_freq_signal_noise[["snr_000", "snr_090", "snr_ver"]]

    snr_smooth = snr.rolling(
        window=config.get_value("window"),
        center=config.get_value("centre"),
        min_periods=config.get_value("min_periods"),
    ).mean()

    # SNR assessment:
    # Need at least min_points_above_thresh points
    # in the interval initial_screening_min_freq_Hz to
    # initial_screening_max_freq_Hz
    # with SNR > initial_screening_snr_thresh_ver
    # for the ver component and SNR > initial_screening_snr_thresh_horiz
    # for the 000 and 090 components

    snr_smooth_freq_interval_for_screening = snr_smooth.loc[
        (
            snr_with_freq_signal_noise["frequency"]
            >= config.get_value("initial_screening_min_freq_Hz")
        )
        & (
            snr_with_freq_signal_noise["frequency"]
            <= config.get_value("initial_screening_max_freq_Hz")
        )
    ]

    num_valid_points_in_interval = np.sum(
        snr_smooth_freq_interval_for_screening
        > [
            config.get_value("initial_screening_snr_thresh_horiz"),
            config.get_value("initial_screening_snr_thresh_horiz"),
            config.get_value("initial_screening_snr_thresh_ver"),
        ],
        axis=0,
    )

    if (
        num_valid_points_in_interval
        > config.get_value("initial_screening_min_points_above_thresh")
    ).all():
        # keep the record
        skipped_record = None

        min_freq = (
            config.get_value("min_freq_Hz")
            if search_start_freq is None
            else search_start_freq
        )
        above_min_freq = snr_with_freq_signal_noise["frequency"] > min_freq
        snr_smooth_gtr_min_freq = snr_smooth.loc[above_min_freq]
        freq_gtr_min_freq = (
            snr_with_freq_signal_noise["frequency"].loc[above_min_freq].to_numpy()
        )

        calculate_fmax_partial = functools.partial(
            calculate_fmax,
            config.get_value("snr_thresh"),
            scaled_nyquist_freq,
            freq_gtr_min_freq,
        )

        fmax_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "fmax_000": calculate_fmax_partial(
                        snr_smooth_gtr_min_freq["snr_000"]
                    ),
                    "fmax_090": calculate_fmax_partial(
                        snr_smooth_gtr_min_freq["snr_090"]
                    ),
                    "fmax_ver": calculate_fmax_partial(
                        snr_smooth_gtr_min_freq["snr_ver"]
                    ),
                    "fmax_status": FmaxStatus.COMPUTED.value,
                    "lpf_default": scaled_nyquist_freq,
                }
            ]
        )

    else:
        # Not enough signal to find an fmax
        fmax_record = nan_fmax_record(
            record_id, FmaxStatus.SCREENING_FAILED, scaled_nyquist_freq
        )

        skipped_record = pd.DataFrame(
            [
                {
                    "record_id": record_id,
                    "reason": (
                        f"Record skipped in fmax calculation because there were not at least "
                        f"{config.get_value('initial_screening_min_points_above_thresh')} frequency points in the interval "
                        f"{config.get_value('initial_screening_min_freq_Hz')} - "
                        f"{config.get_value('initial_screening_max_freq_Hz')} Hz "
                        f"with SNR > {config.get_value('initial_screening_snr_thresh_ver')} for the ver component and "
                        f"SNR > {config.get_value('initial_screening_snr_thresh_horiz')} for the 000 and 090 components."
                    ),
                }
            ]
        )

    return fmax_record, skipped_record


def calculate_fmax(
    snr_thresh: float,
    scaled_nyquist_freq: float,
    freq: np.ndarray,
    snr: np.ndarray,
) -> float:
    """
    Calculates the maximum usable frequency (fmax).

    Parameters
    ----------
    snr_thresh : float
        The SNR threshold for data
        points to be considered.
    scaled_nyquist_freq : float
        The Nyquist frequency scaled by
        a factor of nyquist_freq_scaling_factor
        in the config file.
    freq : np.ndarray
        1D array of frequency values.
    snr : np.ndarray
        1D of SNR values.

    Returns
    -------
    float
        Maximum frequency.

    """

    loc = np.where(snr < snr_thresh)[0]

    if len(loc) > 0:
        fmax = min(freq[loc[0]], scaled_nyquist_freq)
    else:
        fmax = min(freq[-1], scaled_nyquist_freq)

    return fmax
