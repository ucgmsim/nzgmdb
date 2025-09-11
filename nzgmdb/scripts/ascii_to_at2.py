"""
Convert a DataFrame of records and corresponding ascii files to PEER AT2 files.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from IM import waveform_reading
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


def write_at2(
    record_id: str,
    ev_datetime: datetime,
    comp: str,
    dt: float,
    values: np.ndarray,
    output_filename: Path,
):
    """
    Write a PEER AT2 file (NGA-Subduction style) from metadata and ground motion values.

    Parameters
    ----------
    record_id : str
        Unique identifier for the record, used in the filename. e.g. "1476956_MSZS_HN_20"
    ev_datetime : datetime
        Event date and time.
    comp : str
        The full component code, e.g. "000", "090", "ver".
    dt : float
        Time step in seconds.
    values : np.ndarray
        Array of ground motion values (in g).
    output_filename : Path
        Output filename for the AT2 file. e.g. "1476956_MSZS_HN_20_000.AT2"
    """
    # Split the record_id into components
    evid, station, _, _ = record_id.split("_")
    npts = len(values)

    with open(output_filename, "w") as f:
        # Header lines
        f.write("NEW ZEALAND GROUND MOTION DATABASE RECORD\n")
        f.write(f"{evid}, {ev_datetime}, {station}, {comp}, {output_filename.name}\n")
        f.write("ACCELERATION TIME SERIES IN UNITS OF G\n")
        f.write(f"NPTS= {npts:6d}, DT= {dt:8.5f} SEC\n")

        # Write data: 5 values per line, scientific notation, width 14.7E
        for i in range(0, npts, 5):
            line_vals = values[i : i + 5]
            f.write(" ".join(f"{v:14.7E}" for v in line_vals) + "\n")


def convert_row_to_at2(
    row: pd.Series, ascii_000: Path, ascii_090: Path, ascii_ver: Path, output_dir: Path
):
    """
    Convert a DataFrame row and ascii files to PEER AT2 files.

    Parameters
    ----------
    row : pd.Series
        A pandas Series containing the record metadata.
        Expected columns: 'record_id', 'datetime'
    ascii_000 : Path
        Path to the ASCII file for the 000 component.
    ascii_090 : Path
        Path to the ASCII file for the 090 component.
    ascii_ver : Path
        Path to the ASCII file for the vertical component.
    output_dir : Path
        Directory to save the output AT2 file.

    Raises
    ------
    ValueError
        If required columns are missing in the input row.
    """
    # Check if required columns are present
    required_columns = ["record_id", "datetime"]
    for col in required_columns:
        if col not in row:
            raise ValueError(f"Missing required column '{col}' in the input row.")

    record_id = row["record_id"]
    ev_datetime = pd.to_datetime(row["datetime"])

    dt, waveform = waveform_reading.read_ascii(ascii_000, ascii_090, ascii_ver)

    components = {
        "000": waveform[0, :, 0],
        "090": waveform[0, :, 1],
        "ver": waveform[0, :, 2],
    }
    for comp, values in components.items():
        output_filename = output_dir / f"{record_id}_{comp}.AT2"
        write_at2(record_id, ev_datetime, comp, dt, values, output_filename)


@cli.from_docstring(app)
def convert_ascii_to_at2(
    df_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
        ),
    ],
    ascii_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
        ),
    ],
):
    """
    Convert a DataFrame of records and corresponding ascii files to PEER AT2 files.

    Parameters
    ----------
    df_path : Path
        Full file path to a pandas DataFrame containing the records metadata.
        Expected columns: 'record_id', 'datetime'
    ascii_dir : Path
        Directory containing the ASCII files.
    output_dir : Path
        Directory to save the output AT2 files.

    Raises
    ------
    ValueError
        If required columns are missing in the input DataFrame.
    """
    df = pd.read_csv(df_path)
    # Check if required columns are present
    required_columns = ["record_id", "datetime"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in the input DataFrame.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Make a dictionary of the ascii files after using rglob to find them
    ascii_files = {
        f.stem: {"000": f, "090": f.with_suffix(".090"), "ver": f.with_suffix(".ver")}
        for f in Path(ascii_dir).rglob("*.000")
    }

    for _, row in df.iterrows():
        record_id = row["record_id"]

        if record_id not in ascii_files:
            print(
                f"Warning: No ASCII files found for record_id '{record_id}'. Skipping."
            )
            continue

        ascii_000 = ascii_files[record_id]["000"]
        ascii_090 = ascii_files[record_id]["090"]
        ascii_ver = ascii_files[record_id]["ver"]

        convert_row_to_at2(row, ascii_000, ascii_090, ascii_ver, output_dir)
