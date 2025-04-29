"""
Script to run the GMC processing for the NZGMDB.
"""

import functools
import multiprocessing
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from nzgmdb.management import file_structure, shell_commands
from qcore import cli

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
    prob_series_ffp: Path,
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
    prob_series_ffp : Path
        The full file path to the prob_series hdf5 file.

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
            features_command = f"python {gmc_scripts_path}/extract_features.py {gmc_dir} {waveform_dir} mseed --ko_matrices_dir {ko_matrices_dir} --record_list_ffp {batch_txt} --phase_arrival_table {phase_arrival_table_ffp} --prob_series {prob_series_ffp}"
            shell_commands.run_command(
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
        shell_commands.run_command(
            predict_command, conda_sh, gmc_predict_activate, log_file_path_predict
        )

        print(f"Successfully processed Batch {batch_num}")
    except Exception as e:  # noqa: BLE001
        print(f"Failed to process Batch {batch_num}: {str(e)}")


@cli.from_docstring(app)
def run_gmc_processing(
    main_dir: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
        ),
    ],
    gm_classifier_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    ko_matrices_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    conda_sh: Annotated[
        Path,
        typer.Argument(),
    ],
    gmc_activate: Annotated[
        str,
        typer.Argument(),
    ],
    gmc_predict_activate: Annotated[
        str,
        typer.Argument(),
    ],
    n_procs: Annotated[
        int,
        typer.Option(),
    ] = 1,
    waveform_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
        ),
    ] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
        ),
    ] = None,
    bypass_records_ffp: Annotated[
        Path,
        typer.Option(),
    ] = None,
):
    """
    Run GMC processing for the NZGMDB pipeline.

    This function processes waveform data using the GMC classifier, applying KO matrices
    and generating GMC predictions.

    Parameters
    ----------
    main_dir : Path
        The main directory for gmdb.
    gm_classifier_dir : Path
        Directory for gm_classifier.
    ko_matrices_dir : Path
        Directory for KO matrices.
    conda_sh : Path
        Path to activate your mamba conda.sh script.
    gmc_activate : str
        Command to activate the GMC environment for extracting features.
    gmc_predict_activate : str
        Command to activate the GMC predict environment to run predictions.
    n_procs : int, optional
        Number of processes to use for multiprocessing (default is 1).
    waveform_dir : Path, optional
        Directory containing all waveform files.
    output_dir : Path, optional
        Output directory for the GMC predictions.
    bypass_records_ffp : Path, optional
        The full file path to the bypass records file, which includes a custom fmin.
    """
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
    prob_series_ffp = flatfile_dir / file_structure.PreFlatfileNames.PROB_SERIES

    # Get all the mseed files
    mseed_files = list(waveform_dir.rglob("*.mseed"))

    # Ensure that n_procs is equal too or less than the number of mseed files
    n_procs = min(n_procs, len(mseed_files))

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
        prob_series_ffp=prob_series_ffp,
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

    # Check if the bypass records file exists
    if bypass_records_ffp is not None:
        bypass_df = pd.read_csv(bypass_records_ffp)
        # Merge in the bypass fmin values
        combined_df = combined_df.merge(
            bypass_df[["record_id", "fmin_000", "fmin_090", "fmin_ver"]],
            left_on="record",
            right_on="record_id",
            how="left",
            suffixes=("", "_bypass"),
        )
        # Replace the fmin values with the bypass values if they are not nan
        mask_000 = combined_df["component"] == "X"
        combined_df.loc[mask_000, "fmin_mean"] = combined_df.loc[
            mask_000, "fmin_000"
        ].fillna(combined_df.loc[mask_000, "fmin_mean"])
        mask_090 = combined_df["component"] == "Y"
        combined_df.loc[mask_090, "fmin_mean"] = combined_df.loc[
            mask_090, "fmin_090"
        ].fillna(combined_df.loc[mask_090, "fmin_mean"])
        mask_ver = combined_df["component"] == "Z"
        combined_df.loc[mask_ver, "fmin_mean"] = combined_df.loc[
            mask_ver, "fmin_ver"
        ].fillna(combined_df.loc[mask_ver, "fmin_mean"])

        # Remove the fmin_000, fmin_090, fmin_ver, record_id_bypass columns
        combined_df = combined_df.drop(
            columns=["fmin_000", "fmin_090", "fmin_ver", "record_id_bypass"]
        )

    combined_df.to_csv(final_predictions_output, index=False)


if __name__ == "__main__":
    app()
