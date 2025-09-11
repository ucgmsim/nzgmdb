import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime

from IM import waveform_reading


def write_at2(record_id: str, ev_datetime: datetime, comp: str, dt: float, values: np.ndarray, output_filename: str):
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
    output_filename : str
        Output filename for the AT2 file. e.g. "1476956_MSZS_HN_20_000.AT2"
    """
    # Split the record_id into components
    evid, station, _, _ = record_id.split("_")
    npts = len(values)

    with open(output_filename, "w") as f:
        # Header lines
        f.write("NEW ZEALAND GROUND MOTION DATABASE RECORD\n")
        f.write(f"{evid}, {ev_datetime}, {station}, {comp}, {output_filename}\n")
        f.write("ACCELERATION TIME SERIES IN UNITS OF G\n")
        f.write(f"NPTS= {npts:6d}, DT= {dt:8.5f} SEC\n")

        # Write data: 5 values per line, scientific notation, width 14.7E
        for i in range(0, npts, 5):
            line_vals = values[i:i+5]
            f.write(" ".join(f"{v:14.7E}" for v in line_vals) + "\n")


def convert_row_to_at2(row: pd.Series, ascii_000: Path, ascii_090: Path, ascii_ver: Path, output_dir: str):
    """
    Convert a DataFrame row and ascii files to PEER AT2 files.

    Parameters
    ----------
    row : pd.Series
        A pandas Series containing the record metadata.
        Expected columns: 'record_id', 'datetime'
    output_dir : str
        Directory to save the output AT2 file.

    Raises
    ------
    ValueError
        If required columns are missing in the input row.
    """
    # Check if required columns are present
    required_columns = ['record_id', 'datetime']
    for col in required_columns:
        if col not in row:
            raise ValueError(f"Missing required column '{col}' in the input row.")
    record_id = row['record_id']
    ev_datetime = pd.to_datetime(row['datetime'])

    dt, waveform = waveform_reading.read_ascii(ascii_000, ascii_090, ascii_ver)

    output_filename = f"{output_dir}/{record_id}_{comp}.AT2"
    write_at2(record_id, ev_datetime, comp, dt, values, output_filename)