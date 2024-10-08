import functools
import multiprocessing
import subprocess
from pathlib import Path
from typing import Annotated, List

import pandas as pd
import typer

from nzgmdb.management import file_structure

app = typer.Typer()


def run_command(
    command: str, env_sh: Path, env_activate_command: str, log_file_path: Path
):
    """
    Run a shell command with optional Conda environment activation.

    Parameters
    ----------
    command : str
        The command to run.
    env_sh : Path
        The path to the conda.sh script.
    env_activate_command : str
        The command to activate the conda environment needed.
    log_file_path : Path
        The path to the log file.
    """
    with open(log_file_path, "w") as log_file:
        # Create the command to source conda.sh, activate the environment, and execute the full command
        command = f"source {env_sh} && {env_activate_command} && {command}"
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=log_file,
            shell=True,
            executable="/bin/bash",
        )
        # Ensure we wait for the process to finish
        process.communicate()
        if process.returncode != 0:
            raise Exception(f"Command failed please check logs in {log_file_path}")


def process_batch(
    mseed_batch: List[Path],
    gmc_dir: Path,
    waveform_dir: Path,
    gmc_scripts_path: Path,
    ko_matrices_dir: Path,
    conda_sh: Path,
    gmc_activate: str,
    gmc_predict_activate: str,
):
    """
    Process a single subfolder: extract features and run predictions.

    Parameters
    ----------
    mseed_batch : List[Path]
        The list of mseed files to process.
    gmc_dir : Path
        The directory to store the GMC results for the current batch.
    waveform_dir : Path
        The directory containing all waveform files.
    gmc_scripts_path : Path
        The path to the gm_classifier scripts directory.
    ko_matrices_dir : Path
        The directory containing the KO matrices.
    conda_sh : Path
        The path to the conda.sh script. (Used to activate the conda GMC environments)
    gmc_activate : str
        The command to activate the GMC environment for extracting features.
    gmc_predict_activate : str
        The command to activate the GMC prediction environment.
    """
    batch_num = gmc_dir.name.split("_")[-1]
    gmc_dir.mkdir(exist_ok=True, parents=True)
    try:
        # Create a txt file with all the mseed files in the batch to process
        batch_txt = gmc_dir / f"batch_{batch_num}.txt"
        with open(batch_txt, "w") as f:
            for mseed_file in mseed_batch:
                f.write(f"{mseed_file.stem}\n")

        # Construct paths for output
        predictions_output = gmc_dir / file_structure.FlatfileNames.GMC_PREDICTIONS
        log_file_path_features = gmc_dir / "extract_features.log"
        log_file_path_predict = gmc_dir / "predict.log"

        # Check if the failed_records directory exists
        if any(f.is_dir() for f in gmc_dir.glob("failed_records*")):
            print(
                f"Skipping extract features for Batch {batch_num} as features already exist"
            )
        else:
            # Activate gmc environment and extract features for the subfolder
            features_command = f"python {gmc_scripts_path}/extract_features.py {gmc_dir} {waveform_dir} mseed --ko_matrices_dir {ko_matrices_dir} --record_list_ffp {batch_txt}"
            run_command(
                features_command, conda_sh, gmc_activate, log_file_path_features
            )

            # Check again that the failed_records directory exists
            if not any(f.is_dir() for f in gmc_dir.glob("failed_records*")):
                raise FileNotFoundError(
                    f"Failed to extract features for Batch {batch_num}. Please check logs in this folder or try a re-run"
                )

        # Check if the gmc_predictions.csv file already exists
        if predictions_output.exists():
            print(f"Skipping Batch {batch_num} as gmc_predictions.csv already exists")
            return

        # Activate gmc_predict environment and run prediction
        predict_command = (
            f"python {gmc_scripts_path}/predict.py {gmc_dir} {predictions_output}"
        )
        run_command(
            predict_command, conda_sh, gmc_predict_activate, log_file_path_predict
        )

        print(f"Successfully processed Batch {batch_num}")
    except Exception as e:
        print(f"Failed to process Batch {batch_num}: {str(e)}")


@app.command()
def run_gmc_processing(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="Main directory for gmdb.",
        ),
    ],
    gm_classifier_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory for gm_classifier.",
        ),
    ],
    ko_matrices_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory for KO matrices.",
        ),
    ],
    conda_sh: Annotated[
        Path,
        typer.Argument(
            help="Path to activate your mamba conda.sh script.",
        ),
    ],
    gmc_activate: Annotated[
        str,
        typer.Argument(
            help="Command to activate gmc environment for extracting features.",
        ),
    ],
    gmc_predict_activate: Annotated[
        str,
        typer.Argument(
            help="Command to activate gmc_predict environment to run the predictions.",
        ),
    ],
    n_procs: Annotated[
        int,
        typer.Option(
            help="Number of processes to use for multiprocessing.",
        ),
    ] = 1,
):
    # Obtain other paths
    gmc_dir = file_structure.get_gmc_dir(main_dir)
    gmc_dir.mkdir(exist_ok=True, parents=True)
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    gmc_scripts_path = gm_classifier_dir / "gm_classifier/scripts"
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    final_predictions_output = (
        flatfile_dir / file_structure.FlatfileNames.GMC_PREDICTIONS
    )

    # Get all subfolders in the waveform directory (every year)
    # subfolders = [f for f in waveform_dir.iterdir() if f.is_dir()]

    # Get all the mseed files
    mseed_files = list(waveform_dir.rglob("*.mseed"))
    # Split them into even batches based on number of mseeds and n_procs
    batch_size = len(mseed_files) // n_procs
    remainder = len(mseed_files) % n_procs
    mseed_batches = []
    start = 0
    for i in range(n_procs):
        end = start + batch_size + (1 if i < remainder else 0)
        mseed_batches.append(mseed_files[start:end])
        start = end

    # Create a partial function with common arguments pre-filled
    process_partial = functools.partial(
        process_batch,
        waveform_dir=waveform_dir,
        gmc_scripts_path=gmc_scripts_path,
        ko_matrices_dir=ko_matrices_dir,
        conda_sh=conda_sh,
        gmc_activate=gmc_activate,
        gmc_predict_activate=gmc_predict_activate,
    )

    # Use multiprocessing with starmap and the partial function
    with multiprocessing.Pool(n_procs) as p:
        p.starmap(
            process_partial,
            [
                (batch, (gmc_dir / f"batch_{idx}"))
                for idx, batch in enumerate(mseed_batches)
            ],
        )

    # For each subfolder combine the gmc_predictions.csv into a single file
    dfs = []
    for gmc_subfolder in gmc_dir.iterdir():
        predictions_output = (
            gmc_subfolder / file_structure.FlatfileNames.GMC_PREDICTIONS
        )
        if predictions_output.exists():
            df = pd.read_csv(predictions_output)
            dfs.append(df)
        else:
            raise FileNotFoundError(
                f"Failed to find {predictions_output} for {gmc_subfolder}. Please check logs in this folder or try a re-run"
            )
    combined_df = pd.concat(dfs)
    combined_df.to_csv(final_predictions_output, index=False)


if __name__ == "__main__":
    app()
