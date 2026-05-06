"""
Contains functions for generating
the phase arrival table
"""

import functools
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from nzgmdb.management import file_structure, shell_commands


def process_batch(
    batches: tuple[list[Path], Path],
    run_phasenet_script_ffp: Path,
    conda_sh: Path,
    env_activate_command: str,
    bypass_records_ffp: Path | None = None,
    xml_dir: Path | None = None,
):
    """
    Process a single subfolder: run PhaseNet over mseeds.

    Parameters
    ----------
    batches : tuple[list[Path], Path]
        Holds the list of mseed files to process and the output directory.
    run_phasenet_script_ffp : Path
        The script full file path to run PhaseNet (In NZGMDB/phase_arrival).
    conda_sh : Path
        The path to the conda.sh script. (Used to activate the conda PhaseNet environment)
    env_activate_command : str
        The command to activate the environment for running PhaseNet.
    bypass_records_ffp : Path
        The full file path to the bypass records file, which includes a custom p_wave_datetime and/or s_wave_datetime
    xml_dir: Path
        The path to the station xml files. Used for reducing FDSN calls that require station information.

    Raises
    ------
    FileNotFoundError
        If the output phase arrival table is not found.
    """
    mseed_batch, output_dir = batches
    output_dir.mkdir(exist_ok=True, parents=True)
    batch_num = output_dir.name.split("_")[-1]

    # Create a txt file with all the mseed files in the batch to process
    batch_txt = output_dir / f"batch_{batch_num}.txt"
    with open(batch_txt, "w") as f:
        for mseed_file in mseed_batch:
            f.write(f"{mseed_file}\n")

    log_file_path_phasenet = output_dir / "run_phasenet.log"

    # Check if the output phase_arrival_table already exists
    if (output_dir / file_structure.FlatfileNames.PHASE_ARRIVAL_TABLE).exists():
        print(f"Skipping run_phasenet for Batch {batch_num} as results already exist")
    else:
        # Activate phaseNet environment and run over mseeds for the subfolder
        phasenet_command = f"python {run_phasenet_script_ffp} {batch_txt} {output_dir}"
        if bypass_records_ffp is not None:
            phasenet_command += f" --bypass_ffp {bypass_records_ffp}"
        if xml_dir is not None:
            phasenet_command += f" --xml_dir {xml_dir}"
        shell_commands.run_command(
            phasenet_command, conda_sh, env_activate_command, log_file_path_phasenet
        )

        # Check again that the output phase_arrival_table exists
        if not (output_dir / file_structure.FlatfileNames.PHASE_ARRIVAL_TABLE).exists():
            raise FileNotFoundError(
                f"Failed to run_phasenet for Batch {batch_num}. Please check logs in this folder or try a re-run"
            )


def generate_phase_arrival_table(
    main_dir: Path,
    run_phasenet_script_ffp: Path,
    conda_sh: Path,
    env_activate_command: str,
    n_procs: int,
    n_batches: int | None = None,
    bypass_records_ffp: Path | None = None,
    xml_dir: Path | None = None,
):
    """
    Generate the phase arrival table utilizing phaseNet

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
        (glob is used to find all mseed files recursively)
    run_phasenet_script_ffp : Path
        The script full file path to run PhaseNet (In NZGMDB/phase_arrival).
    conda_sh : Path
        The path to the conda.sh script. (Used to activate the conda PhaseNet environment)
    env_activate_command : str
        The command to activate the environment for running PhaseNet.
    n_procs : int
        The number of processes to use
    n_batches : int, optional
        The number of batches to split the mseed files into. If None, it will be set to the number of processes. (Default is None)
    bypass_records_ffp : Path
        The full file path to the bypass records file, which includes a custom p_wave_ix
    xml_dir: Path
        The path to the station xml files. Used for reducing FDSN calls that require station information.
    """
    # Get the Phase_arrival directory
    phase_dir = main_dir / "phase_arrival"
    phase_dir.mkdir(exist_ok=True)

    # Get the flatfile directory
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)

    # Find all mseed files recursively
    mseed_files = list(main_dir.rglob("*.mseed"))

    # Split them into even batches based on number of mseeds and n_procs
    # Ensure n_procs and n_batches gets reduced if it is greater than the number of mseed files
    n_procs = min(n_procs, len(mseed_files))
    n_batches = n_batches or n_procs
    n_batches = min(n_batches, len(mseed_files))
    mseed_batches = np.array_split(mseed_files, n_batches)

    batches = [
        (batch, (phase_dir / f"batch_{idx}")) for idx, batch in enumerate(mseed_batches)
    ]

    # Checkpointing: only schedule batches that are missing outputs
    pending_batches = []
    for batch, out_dir in batches:
        phase_table_ffp = out_dir / file_structure.FlatfileNames.PHASE_ARRIVAL_TABLE
        if phase_table_ffp.exists():
            batch_num = out_dir.name.split("_")[-1]
            print(f"Skipping Batch {batch_num} (found existing phase arrival table)")
            continue
        pending_batches.append((batch, out_dir))

    if not pending_batches:
        print("All batches already have a phase arrival table; nothing to run.")
    else:
        # Fetch results (only for pending batches)
        with mp.Pool(n_procs) as p:
            p.map(
                functools.partial(
                    process_batch,
                    run_phasenet_script_ffp=run_phasenet_script_ffp,
                    conda_sh=conda_sh,
                    env_activate_command=env_activate_command,
                    bypass_records_ffp=bypass_records_ffp,
                    xml_dir=xml_dir,
                ),
                pending_batches,
            )

    # For each subfolder combine the phase_arrival_table.csv and skipped_records.csv into a single file
    phase_results = []
    skipped_records_results = []
    prob_series_files = []
    for phase_subfolder in phase_dir.iterdir():
        phase_output = (
            phase_subfolder / file_structure.FlatfileNames.PHASE_ARRIVAL_TABLE
        )
        skipped_output = phase_subfolder / "skipped_records.csv"
        prob_series_output = phase_subfolder / "prob_series.h5"
        if phase_output.exists():
            df = pd.read_csv(phase_output)
            phase_results.append(df)
        else:
            raise FileNotFoundError(
                f"Failed to find {phase_output} for {phase_subfolder}. Please check logs in this folder or try a re-run"
            )
        if skipped_output.exists():
            df = pd.read_csv(skipped_output)
            skipped_records_results.append(df)
        else:
            raise FileNotFoundError(
                f"Failed to find {skipped_output} for {phase_subfolder}. Please check logs in this folder or try a re-run"
            )
        if prob_series_output.exists():
            prob_series_files.append(prob_series_output)
        else:
            raise FileNotFoundError(
                f"Failed to find {prob_series_output} for {phase_subfolder}. Please check logs in this folder or try a re-run"
            )

    # Merge the prob_series files
    prob_series_output_ffp = flatfile_dir / file_structure.PreFlatfileNames.PROB_SERIES
    with h5py.File(prob_series_output_ffp, "w") as out_f:
        for prob_series_file in prob_series_files:
            with h5py.File(prob_series_file, "r") as in_f:
                for record_name in in_f.keys():
                    in_f.copy(record_name, out_f)

    # Concatenate the results
    phase_df = pd.concat(phase_results)
    skipped_df = pd.concat(skipped_records_results)

    # Ensure the p_wave_ix and s_wave_ix column is an int
    phase_df["p_wave_ix"] = phase_df["p_wave_ix"].astype(int)
    phase_df["s_wave_ix"] = phase_df["s_wave_ix"].astype(int)

    # Load the earthquake source table to add evid_datetime
    source_table = pd.read_csv(
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_TECTONIC,
        dtype={"evid": str},
    )

    # Create the evid col in the phase table with the split of record_id taking the 1st index
    phase_df["evid"] = phase_df["record_id"].str.split("_").str[0]

    # Merge in the datetime from the source table
    phase_df = phase_df.merge(
        source_table[["evid", "datetime"]],
        on="evid",
        how="left",
    )

    # Relabel the datetime column to evid_datetime
    phase_df = phase_df.rename(columns={"datetime": "evid_datetime"})

    # Remove the evid column
    phase_df = phase_df.drop(columns=["evid"])

    # Save the phase arrival table
    phase_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE, index=False
    )
    skipped_df.to_csv(
        flatfile_dir
        / file_structure.SkippedRecordFilenames.PHASE_ARRIVAL_SKIPPED_RECORDS,
        index=False,
    )
