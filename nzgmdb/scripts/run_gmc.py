import functools
import multiprocessing
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from nzgmdb.management import commands, file_structure

app = typer.Typer()


def process_batch(
    mseed_batch: list[Path],
    gmc_dir: Path,
    waveform_dir: Path,
    gmc_scripts_path: Path,
    ko_matrices_dir: Path,
    conda_sh: Path,
    gmc_activate: str,
    gmc_predict_activate: str,
    phase_arrival_table_ffp: Path,
):
    """
    Process a single subfolder: extract features and run predictions.

    Parameters
    ----------
    mseed_batch : list[Path]
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
    phase_arrival_table_ffp : Path
        The full file path to the phase arrival table

    Raises
    ------
    Exception
        If feature extraction or predict fails.
    FileNotFoundError
        If the failed records dir is not found.
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
            features_command = f"python {gmc_scripts_path}/extract_features.py {gmc_dir} {waveform_dir} mseed --ko_matrices_dir {ko_matrices_dir} --record_list_ffp {batch_txt} --phase_arrival_table {phase_arrival_table_ffp}"
            commands.run_command(
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
        commands.run_command(
            predict_command, conda_sh, gmc_predict_activate, log_file_path_predict
        )

        print(f"Successfully processed Batch {batch_num}")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to process Batch {batch_num}: {str(e)}")


@app.command()
def run_gmc_processing(  # noqa: D103
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="Main directory for gmdb.",
            file_okay=False,
        ),
    ],
    gm_classifier_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory for gm_classifier.",
            exists=True,
            file_okay=False,
        ),
    ],
    ko_matrices_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory for KO matrices.",
            exists=True,
            file_okay=False,
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
    waveform_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing all waveform files.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            help="Output directory for the GMC predictions.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
):
    # Obtain other paths
    gmc_dir = file_structure.get_gmc_dir(main_dir)
    gmc_dir.mkdir(exist_ok=True, parents=True)
    waveform_dir = (
        file_structure.get_waveform_dir(main_dir)
        if waveform_dir is None
        else waveform_dir
    )
    gmc_scripts_path = gm_classifier_dir / "gm_classifier/scripts"
    output_dir = (
        file_structure.get_flatfile_dir(main_dir) if output_dir is None else output_dir
    )
    final_predictions_output = output_dir / file_structure.FlatfileNames.GMC_PREDICTIONS

    # Get the phase arrival table
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    phase_arrival_table_ffp = (
        flatfile_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
    )

    # Get all the mseed files
    mseed_files = list(waveform_dir.rglob("*.mseed"))

    # Split them into even batches based on number of mseeds and n_procs
    mseed_batches = np.array_split(mseed_files, n_procs)

    # Create a partial function with common arguments pre-filled
    process_partial = functools.partial(
        process_batch,
        waveform_dir=waveform_dir,
        gmc_scripts_path=gmc_scripts_path,
        ko_matrices_dir=ko_matrices_dir,
        conda_sh=conda_sh,
        gmc_activate=gmc_activate,
        gmc_predict_activate=gmc_predict_activate,
        phase_arrival_table_ffp=phase_arrival_table_ffp,
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
