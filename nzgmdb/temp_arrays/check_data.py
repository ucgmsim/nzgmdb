from pathlib import Path

import pandas as pd
import typer
from obspy import UTCDateTime

app = typer.Typer(pretty_exceptions_enable=False)

# Month length in seconds (30 days)
MONTH_SECONDS = 30 * 24 * 3600


def _format_end_for_filename(end_dt: UTCDateTime | str) -> str:
    """Format an end datetime into the MiniSEED filename timestamp form.

    Parameters
    ----------
    end_dt : UTCDateTime | str
        An ObsPy ``UTCDateTime`` or any string parseable by ``UTCDateTime``.

    Returns
    -------
    str
        Timestamp string formatted like ``YYYYMMDDTHHMMSSZ``.

    Raises
    ------
    ValueError
        If ``end_dt`` cannot be parsed by ``UTCDateTime``.
    """
    if not isinstance(end_dt, UTCDateTime):
        end_dt = UTCDateTime(end_dt)
    return end_dt.strftime("%Y%m%dT%H%M%SZ")


def _row_mseed_dir(output_dir: Path, mseed_dirname: str, row: pd.Series) -> Path:
    """Build the expected output MiniSEED directory for a CSV row.

    Parameters
    ----------
    output_dir : Path
        Root output directory containing the MiniSEED subdirectory.
    mseed_dirname : str
        Name of the MiniSEED subdirectory (typically ``"waveforms"``).
    row : pandas.Series
        A row from the input CSV containing at least ``net``, ``sta``, ``loc``,
        and ``chan``.

    Returns
    -------
    Path
        Path like ``<output_dir>/<mseed_dirname>/<net>/<net>_<sta>_<chan>_<loc>``.
    """
    net = str(row["net"]).strip()
    sta = str(row["sta"]).strip()
    loc_field = str(row["loc"])
    chan_prefix = str(row["chan"]).strip()

    record_sub = f"{net}_{sta}_{chan_prefix}_{loc_field}"
    return output_dir / mseed_dirname / net / record_sub


def is_row_done(row: pd.Series, *, output_dir: Path, mseed_dirname: str) -> bool:
    """Check whether a CSV row has fully completed waveform download.

    A row is considered **done** if the expected MiniSEED output directory
    contains at least one ``.mseed`` file whose final ``__END`` timestamp equals
    the row's ``end_date``.

    Notes
    -----
    This check is filesystem-driven and intentionally conservative:

    - Only files whose channel code ends with ``"Z"`` are considered.
    - Any filesystem error or unexpected filename format causes this function
      to return ``False`` (so the row can be retried).

    Parameters
    ----------
    row : pandas.Series
        A row of the input CSV.
    output_dir : Path
        Root output directory.
    mseed_dirname : str
        Name of the MiniSEED subdirectory (typically ``"waveforms"``).

    Returns
    -------
    bool
        ``True`` if the row appears fully completed, otherwise ``False``.
    """
    mseed_path = _row_mseed_dir(output_dir, mseed_dirname, row)

    if not mseed_path.is_dir():
        return False

    target_end = _format_end_for_filename(row["end_date"])

    try:
        for path in mseed_path.iterdir():
            if path.suffix != ".mseed":
                continue

            stem = path.stem

            # Filename parts expected like: NET.STA..CHAN__START__END
            parts = stem.split(".")
            if len(parts) <= 3:
                continue

            chan_check = parts[3].split("__", 1)[0]  # remove any suffix after __
            if not chan_check.endswith("Z"):
                continue

            if "__" not in stem:
                continue

            last = stem.rsplit("__", 1)[-1]
            if last == target_end:
                return True
    except OSError:
        # Any filesystem error -> treat as not done so row will be retried.
        return False

    return False


def is_row_started(row: pd.Series, *, output_dir: Path, mseed_dirname: str) -> bool:
    """Check whether any waveform data exists for a CSV row.

    Parameters
    ----------
    row : pandas.Series
        A row of the input CSV.
    output_dir : Path
        Root output directory.
    mseed_dirname : str
        Name of the MiniSEED subdirectory (typically ``"waveforms"``).

    Returns
    -------
    bool
        ``True`` if at least one ``.mseed`` file exists for the row.
    """
    mseed_path = _row_mseed_dir(output_dir, mseed_dirname, row)

    if not mseed_path.is_dir():
        return False

    return any(p.suffix == ".mseed" for p in mseed_path.iterdir())


def evaluate_download_completeness(
    csv_file: Path,
    output_csv: Path,
    output_dir: Path,
    provider: str = "IRIS",
) -> pd.DataFrame:
    """Evaluate and summarize download completeness for waveform requests.

    The evaluation is performed at two levels:

    - **completed**: at least one MiniSEED file exists with a final ``__END``
      timestamp equal to the row ``end_date``.
    - **started**: at least one MiniSEED file exists for the row.

    Parameters
    ----------
    csv_file : Path
        Input CSV describing waveform requests.
    output_csv : Path
        Output CSV to write the evaluation results to.
    output_dir : Path
        Root output directory containing the MiniSEED data.
    provider : str, optional
        Provider name to filter by, by default "IRIS".

    Returns
    -------
    pandas.DataFrame
        The evaluated dataframe including the ``completed`` and ``started``
        boolean columns.

    Raises
    ------
    ValueError
        If required columns are missing from the input CSV.
    """
    df = pd.read_csv(csv_file, dtype={"loc": str}, keep_default_na=False)

    required_cols = {
        "provider",
        "net",
        "sta",
        "loc",
        "chan",
        "start_date",
        "end_date",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    df = df[df["provider"] == provider].reset_index(drop=True)

    print(f"Evaluating {len(df)} rows for completeness...")

    mseed_dirname = "waveforms"

    df["completed"] = df.apply(
        is_row_done, axis=1, output_dir=output_dir, mseed_dirname=mseed_dirname
    )
    df["started"] = df.apply(
        is_row_started, axis=1, output_dir=output_dir, mseed_dirname=mseed_dirname
    )

    total = len(df)
    completed = int(df["completed"].sum())
    started = int(df["started"].sum())

    partial = int(((df["started"]) & (~df["completed"])).sum())
    none = int((~df["started"]).sum())

    print("\n===== DOWNLOAD SUMMARY =====")
    print(f"Total rows            : {total}")
    print(f"Fully completed       : {completed}")
    print(f"Started (any data)    : {started}")
    print(f"Partial (started only): {partial}")
    print(f"No data at all        : {none}")

    cols = ["provider", "net", "sta", "loc", "chan", "start_date", "end_date"]

    if none > 0:
        print("\n===== ROWS WITH NO DATA =====")
        print(df.loc[~df["started"], cols].to_string(index=False))

    if partial > 0:
        print("\n===== PARTIALLY DOWNLOADED ROWS =====")
        print(df.loc[df["started"] & ~df["completed"], cols].to_string(index=False))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nWrote evaluation CSV to:\n  {output_csv}")

    return df


@app.command()
def run(
    csv_file: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input waveform request CSV."
    ),
    output_csv: Path = typer.Argument(
        ..., dir_okay=False, help="Output CSV to write evaluation results to."
    ),
    output_dir: Path = typer.Argument(
        ...,
        exists=False,
        file_okay=False,
        help="Root output directory containing the MiniSEED data.",
    ),
    provider: str = typer.Option(
        "IRIS",
        help="Provider name to filter by.",
    ),
) -> None:
    """Evaluate MiniSEED download completeness for a request table.

    Parameters
    ----------
    csv_file : Path
        Input CSV describing waveform requests.
    output_csv : Path
        Output CSV to write the evaluation results.
    output_dir : Path
        Root output directory containing the MiniSEED data.
    provider : str
        Provider name to filter by.

    Returns
    -------
    None
    """
    evaluate_download_completeness(
        csv_file=csv_file,
        output_csv=output_csv,
        output_dir=output_dir,
        provider=provider,
    )


if __name__ == "__main__":
    app()
