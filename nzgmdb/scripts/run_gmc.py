import functools
import multiprocessing
import subprocess
from pathlib import Path

import pandas as pd
import typer

from nzgmdb.management import file_structure

app = typer.Typer()


def run_command(command, env_sh, env_activate_command):
    """Run a shell command with optional Conda environment activation."""
    # Create the command to source conda.sh, activate the environment, and execute the full command
    command = f"source {env_sh} && {env_activate_command} && {command}"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        executable="/bin/bash",
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise Exception(f"Command failed with error: {stderr.decode('utf-8')}")
    return stdout.decode("utf-8")


def process_subfolder(
    subfolder: Path,
    gmc_dir: Path,
    gmc_scripts_path: Path,
    ko_matrices_dir: Path,
    conda_sh: Path,
    gmc_activate: str,
    gmc_predict_activate: str,
):
    """Process a single subfolder: extract features and run predictions."""
    try:
        gmc_subfolder = gmc_dir / subfolder.stem
        gmc_subfolder.mkdir(exist_ok=True, parents=True)
        # Construct paths for output
        predictions_output = gmc_subfolder / "gmc_predictions.csv"

        # Check if features already exist
        if (gmc_subfolder / "features_comp_X.csv").exists():
            print(
                f"Skipping extract features for {subfolder} as features already exist"
            )
        else:
            # Activate gmc environment and extract features for the subfolder
            features_command = f"python {gmc_scripts_path}/extract_features.py {gmc_subfolder} {subfolder} mseed --ko_matrices_dir {ko_matrices_dir}"
            run_command(features_command, conda_sh, gmc_activate)

        # Check if the gmc_predictions.csv file already exists
        if predictions_output.exists():
            print(f"Skipping {subfolder} as gmc_predictions.csv already exists")
            return

        # Activate gmc_predict environment and run prediction
        predict_command = (
            f"python {gmc_scripts_path}/predict.py {gmc_subfolder} {predictions_output}"
        )
        run_command(predict_command, conda_sh, gmc_predict_activate)

        print(f"Successfully processed {subfolder}")
    except Exception as e:
        print(f"Failed to process {subfolder}: {str(e)}")


@app.command()
def run_gmc_processing(
    main_dir: Path = typer.Argument(..., help="Main directory for gmdb."),
    gm_classifier_dir: Path = typer.Argument(..., help="Directory for gm_classifier."),
    ko_matrices_dir: Path = typer.Argument(..., help="Directory for KO matrices."),
    conda_sh: Path = typer.Argument(
        ..., help="Path to activate your mamba conda.sh script."
    ),
    gmc_activate: str = typer.Argument(
        ..., help="Command to activate gmc environment for extracting features."
    ),
    gmc_predict_activate: str = typer.Argument(
        ..., help="Command to activate gmc_predict environment to run the predictions."
    ),
    n_procs: int = typer.Option(
        1, help="Number of processes to use for multiprocessing."
    ),
):
    # Obtain other paths
    gmc_dir = file_structure.get_gmc_dir(main_dir)
    gmc_dir.mkdir(exist_ok=True, parents=True)
    waveform_dir = file_structure.get_waveform_dir(main_dir)
    gmc_scripts_path = gm_classifier_dir / "gm_classifier/scripts"
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    final_predictions_output = flatfile_dir / "gmc_predictions.csv"

    # Get all subfolders in the waveform directory (every year)
    subfolders = [f for f in waveform_dir.iterdir() if f.is_dir()]

    # Create a partial function with common arguments pre-filled
    process_partial = functools.partial(
        process_subfolder,
        gmc_dir=gmc_dir,
        gmc_scripts_path=gmc_scripts_path,
        ko_matrices_dir=ko_matrices_dir,
        conda_sh=conda_sh,
        gmc_activate=gmc_activate,
        gmc_predict_activate=gmc_predict_activate,
    )

    # Use multiprocessing with a regular map and the partial function
    with multiprocessing.Pool(n_procs) as p:
        p.map(process_partial, subfolders)

    # For each subfolder combine the gmc_predictions.csv into a single file
    dfs = []
    for gmc_subfolder in gmc_dir.iterdir():
        predictions_output = gmc_subfolder / "gmc_predictions.csv"
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
