import functools
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from IM import im_calculation, waveform_reading, ims
from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure


def calculate_im_for_record(
    ffp_000: Path,
    output_path: Path,
    intensity_measures: list[ims.IM],
    psa_periods: np.ndarray,
    fas_frequencies: np.ndarray,
    ko_bandwith: int = 40,
):
    """
    Calculate the IMs for a single record and save the results to a csv file

    Parameters
    ----------
    ffp_000 : Path
        The full file path to the 000 component file
    output_path : Path
        The path to the output directory
    intensity_measures : list[ims.IM]
        The list of intensity measures to calculate
    psa_periods : np.ndarray
        The periods for calculating the pseudo-spectral acceleration
    fas_frequencies : np.ndarray
        The frequencies for calculating the Fourier amplitude spectrum
    ko_bandwith : int, optional
        The bandwidth for the Konno-Ohmachi smoothing, by default 40
    """
    # Get the 090 and ver components full file paths
    ffp_090 = ffp_000.parent / f"{ffp_000.stem}.090"
    ffp_ver = ffp_000.parent / f"{ffp_000.stem}.ver"
    record_id = ffp_000.stem

    try:
        dt, waveform = waveform_reading.read_ascii(ffp_000, ffp_090, ffp_ver)
    except FileNotFoundError:
        skipped_record_dict = {
            "record_id": record_id,
            "reason": "Failed to find the waveform ascii files",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Get the event_id and create the output directory
    event_id = file_structure.get_event_id_from_mseed(ffp_000)
    event_output_path = output_path / event_id
    event_output_path.mkdir(exist_ok=True, parents=True)

    nyquist_feq = (1 / dt) * 0.5
    # Filter the fas_frequencies to be less than or equal to the nyquist frequency
    fas_frequencies = fas_frequencies[fas_frequencies <= nyquist_feq]

    # Calculate the IMs
    im_result_df = im_calculation.calculate_ims(
        waveform,
        dt,
        intensity_measures,
        psa_periods,
        fas_frequencies,
        cores=1,
        ko_bandwidth=ko_bandwith,
    )

    # Set a column for the record_id and then component and set at the front
    im_result_df = im_result_df.reset_index()
    im_result_df = im_result_df.rename(columns={"index": "component"})
    im_result_df.insert(0, "record_id", record_id)

    # Save the file
    im_result_df.to_csv(event_output_path / f"{record_id}_IM.csv", index=False)


def compute_ims_for_all_processed_records(
    main_dir: Path,
    output_path: Path,
    n_procs: int = 1,
    checkpoint: bool = False,
):
    """
    Compute the IMs for all processed records in the main directory

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    output_path : Path
        The path to the output directory
    n_procs : int, optional
        The number of processes to use
    checkpoint : bool, optional
        If True, the function will check for already completed files and skip them
    """
    # Get the waveform directory and all the 000 files
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    comp_000_files = list(waveform_dir.rglob("*.000"))

    if checkpoint:
        # Get list of already completed files and remove _IM suffix
        completed_files = [f.stem[:-3] for f in output_path.rglob("*_IM.csv")]
        # Remove completed files from the list
        comp_000_files = [f for f in comp_000_files if f.stem not in completed_files]

    print(f"Calculating IMs for {len(comp_000_files)} records")

    # Load the config and extract the IM options
    config = cfg.Config()
    # intensity_measures = [ims.IM[measure] for measure in config.get_value("ims")]
    intensity_measures = [ims.IM.CAV, ims.IM.AI, ims.IM.Ds575, ims.IM.Ds595, ims.IM.CAV5, ims.IM.PGV]
    psa_periods = np.asarray(config.get_value("psa_periods"))
    fas_frequencies = np.logspace(
        np.log10(config.get_value("common_frequency_start")),
        np.log10(config.get_value("common_frequency_end")),
        num=config.get_value("common_frequency_num"),
    )
    ko_bandwith = config.get_value("ko_bandwidth")

    # Fetch results
    with mp.Pool(n_procs) as p:
        skipped_records = p.map(
            functools.partial(
                calculate_im_for_record,
                output_path=output_path,
                intensity_measures=intensity_measures,
                psa_periods=psa_periods,
                fas_frequencies=fas_frequencies,
                ko_bandwith=ko_bandwith,
            ),
            comp_000_files,
        )

    print("Finished calculating IMs")

    # Save the skipped records
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Check that there are skipped_records dataframes that are not None
    if not all(value is None for value in skipped_records):
        skipped_records_df = pd.concat(skipped_records).reset_index(drop=True)
    else:
        print("No skipped records")
        skipped_records_df = pd.DataFrame(columns=["record_id", "reason"])

    if checkpoint:
        # Add the skipped records to the existing skipped records
        try:
            existing_skipped_records = pd.read_csv(
                flatfile_dir
                / file_structure.SkippedRecordFilenames.IM_CALC_SKIPPED_RECORDS
            )
            skipped_records_df = pd.concat(
                [existing_skipped_records, skipped_records_df]
            ).reset_index(drop=True)
        except FileNotFoundError:
            pass

    skipped_records_df.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.IM_CALC_SKIPPED_RECORDS,
        index=False,
    )
