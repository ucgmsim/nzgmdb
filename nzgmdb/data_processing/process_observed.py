import functools
import glob
import multiprocessing
from pathlib import Path

import obspy
import pandas as pd
import qcore.timeseries as ts

from nzgmdb.data_processing import waveform_manipulation
from nzgmdb.management import file_structure


def process_single_mseed(mseed_file: str, gmc_df: pd.DataFrame, fmax_df: pd.DataFrame):
    """
    Process a single mseed file and save the processed data to a txt file
    Will return a dataframe containing the skipped record name and reason why
    if the record must be skipped due to either not containing 3 components,
    failing to find the inventory information or the lowcut frequency being
    greater than the highcut frequency during processing

    Parameters
    ----------
    mseed_file : str
        The path to the mseed file
    gmc_df : pd.DataFrame
        The GMC values containing fmin information
    fmax_df : pd.DataFrame
        The Fmax values

    Returns
    -------
    skipped_record : pd.DataFrame, None
        Dataframe containing the skipped record name and reason why
        or None if the record was processed successfully
    """
    # Read mseed information
    mseed = obspy.read(mseed_file)
    mseed_stem = Path(mseed_file).stem

    # Extract mseed values
    dt = mseed.traces[0].stats.delta
    station = mseed.traces[0].stats.station

    # Check the length of the mseed file for 3 components
    if len(mseed) != 3:
        skipped_record_dict = {
            "mseed_file": mseed_stem,
            "reason": "File did not contain 3 components",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Perform initial pre-processing
    mseed = waveform_manipulation.initial_preprocessing(mseed)

    if mseed is None:
        skipped_record_dict = {
            "mseed_file": mseed_stem,
            "reason": "Failed to find Inventory information",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Get the GMC fmin and fmax values
    fmin_rows = gmc_df[gmc_df["record"] == mseed_stem]
    fmin = None if fmin_rows.empty else fmin_rows["fmin_mean"].max()
    # TODO: Change the fmax_df to have the same format as the gmc_df for searching
    search_name = "_".join(mseed_stem.split("_")[:-1])
    fmax_rows = fmax_df[fmax_df.ev_sta == search_name]
    fmax = (
        None
        if fmax_rows.empty
        else min(fmax_rows.loc[:, ["fmax_000", "fmax_090", "fmax_ver"]].values[0])
    )

    # Perform high and lowcut processing
    try:
        (
            acc_bb_000,
            acc_bb_090,
            acc_bb_ver,
        ) = waveform_manipulation.high_and_low_cut_processing(mseed, dt, fmin, fmax)
    except ValueError:
        skipped_record_dict = {
            "mseed_file": mseed_stem,
            "reason": "Lowcut frequency is greater than the highcut frequency",
        }
        skipped_record = pd.DataFrame([skipped_record_dict])
        return skipped_record

    # Create the output directory
    output_dir = file_structure.get_processed_dir_from_mseed(Path(mseed_file))
    output_dir.mkdir(exist_ok=True)

    # Write the data to the output directory
    for comp, acc_bb in zip(
        ["000", "090", "ver"], [acc_bb_000, acc_bb_090, acc_bb_ver]
    ):
        filename = output_dir / f"{mseed_stem}.{comp}"
        ts.timeseries_to_text(
            acc_bb,
            filename,
            dt,
            station,
            comp,
        )


def process_mseeds_to_txt(
    main_dir: Path, gmc_ffp: Path, fmax_ffp: Path, n_procs: int = 1
):
    """
    Process the mseed files to txt files
    Saves the skipped records to a csv file and gives reasons why they were skipped

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
    gmc_ffp : Path
        The full file path to the GMC predictions file
    fmax_ffp : Path
        The full file path to the Fmax file
    n_procs : int
        The number of processes to use for multiprocessing
    """
    # Get the raw waveform mseed files
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    mseed_files = glob.glob(f"{waveform_dir}/**/*.mseed", recursive=True)

    # Load the GMC and Fmax files
    gmc_df = pd.read_csv(gmc_ffp)
    fmax_df = pd.read_csv(fmax_ffp)

    # Use multiprocessing to process the mseed files
    with multiprocessing.Pool(processes=n_procs) as pool:
        skipped_records = pool.map(
            functools.partial(
                process_single_mseed,
                gmc_df=gmc_df,
                fmax_df=fmax_df,
            ),
            mseed_files,
        )

    # Combine the skipped records
    skipped_records = pd.concat(skipped_records)

    # Save the skipped records
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    skipped_records.to_csv(flatfile_dir / "processing_skipped_records.csv", index=False)
