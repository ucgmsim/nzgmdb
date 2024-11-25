"""
Contains functions for generating
the phase arrival table
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nzgmdb.management import commands, custom_multiprocess, file_structure


def process_batch(
    batches: tuple[list[Path], Path],
    run_phasenet_script_ffp: Path,
    conda_sh: Path,
    env_activate_command: str,
):
    """
    Process a single subfolder: run PhaseNet over mseeds.

    Parameters
    ----------
    batches : tuple[list[Path], Path]
        Holds the list of mseed files to process and the output directory.
    run_phasenet_script_ffp : Path
        The script full file path to run PhaseNet (In NZGMDB/phase_arrival).
    output_dir : Path
        The directory to save the results.
    conda_sh : Path
        The path to the conda.sh script. (Used to activate the conda PhaseNet environment)
    env_activate_command : str
        The command to activate the environment for running PhaseNet.

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
        commands.run_command(
            phasenet_command, conda_sh, env_activate_command, log_file_path_phasenet
        )

        # Check again that the output phase_arrival_table exists
        if not (output_dir / file_structure.FlatfileNames.PHASE_ARRIVAL_TABLE).exists():
            raise FileNotFoundError(
                f"Failed to run_phasenet for Batch {batch_num}. Please check logs in this folder or try a re-run"
            )


def generate_phase_arrival_table(
    main_dir: Path,
    output_dir: Path,
    run_phasenet_script_ffp: Path,
    conda_sh: Path,
    env_activate_command: str,
    n_procs: int,
):
    """
    Generate the phase arrival table utilizing phaseNet

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (Highest level directory)
        (glob is used to find all mseed files recursively)
    output_dir : Path
        The directory to save the phase arrival table
    run_phasenet_script_ffp : Path
        The script full file path to run PhaseNet (In NZGMDB/phase_arrival).
    conda_sh : Path
        The path to the conda.sh script. (Used to activate the conda PhaseNet environment)
    env_activate_command : str
        The command to activate the environment for running PhaseNet.
    n_procs : int
        The number of processes to use
    """
    # Get the Phase_arrival directory
    phase_dir = main_dir / "phase_arrival"
    phase_dir.mkdir(exist_ok=True)

    # Find all mseed files recursively
    mseed_files = list(main_dir.rglob("*.mseed"))

    # Split them into even batches based on number of mseeds and n_procs
    mseed_batches = np.array_split(mseed_files, n_procs)

    batches = [
        (batch, (phase_dir / f"batch_{idx}")) for idx, batch in enumerate(mseed_batches)
    ]

    custom_multiprocess.custom_multiprocess(
        process_batch,
        batches,
        n_procs,
        run_phasenet_script_ffp,
        conda_sh,
        env_activate_command,
    )

    # For each subfolder combine the phase_arrival_table.csv and skipped_records.csv into a single file
    phase_results = []
    skipped_records_results = []
    for phase_subfolder in phase_dir.iterdir():
        phase_output = (
            phase_subfolder / file_structure.FlatfileNames.PHASE_ARRIVAL_TABLE
        )
        skipped_output = phase_subfolder / "skipped_records.csv"
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

    # Concatenate the results
    phase_df = pd.concat(phase_results)
    skipped_df = pd.concat(skipped_records_results)

    # Save the phase arrival table
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_df.to_csv(
        output_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE, index=False
    )
    skipped_df.to_csv(
        output_dir
        / file_structure.SkippedRecordFilenames.PHASE_ARRIVAL_SKIPPED_RECORDS,
        index=False,
    )
