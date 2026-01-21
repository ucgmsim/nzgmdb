import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import (
    GlobalDomain,
    MassDownloader,
    Restrictions,
)

app = typer.Typer(pretty_exceptions_enable=False)

# Month length in seconds (15 days)
MONTH_SECONDS = 15 * 24 * 3600
# Minimum chunk size (seconds) when backing off after 413 / manifest-too-large.
# 3600s = 1 hour.
MIN_CHUNK_SECONDS: int = 3600


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
    """
    if not isinstance(end_dt, UTCDateTime):
        end_dt = UTCDateTime(end_dt)
    return end_dt.strftime("%Y%m%dT%H%M%SZ")


def save_results(
    results: list[dict[str, object] | object],
    results_csv: Path,
) -> None:
    """Write a single results CSV from per-row download results.

    This function is intentionally defensive:

    - It computes the union of keys across result dictionaries.
    - It normalizes missing keys to empty strings.
    - It serializes nested objects (dict/list) as JSON for readability.

    Parameters
    ----------
    results : list[dict[str, object] | object]
        Download results. Most entries are dicts, but non-dicts will be written
        under a ``value`` column.
    results_csv : Path
        Output CSV file.
    """
    if not results:
        results_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(results_csv, index=False)
        return

    all_keys: set[str] = set()
    for r in results:
        if isinstance(r, dict):
            all_keys.update(r.keys())
        else:
            all_keys.add("value")

    preferred = [
        "idx",
        "provider",
        "net",
        "sta",
        "status",
        "error",
        "attempts",
        "chunklength",
        "timestamp",
        "mseed_path",
        "xml_path",
    ]
    cols = [k for k in preferred if k in all_keys] + sorted(
        k for k in all_keys if k not in preferred
    )

    norm_rows: list[dict[str, object]] = []
    for r in results:
        if not isinstance(r, dict):
            row: dict[str, object] = {"value": str(r)}
            norm_rows.append(row)
            continue

        row = {}
        for k in cols:
            v = r.get(k, "")
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            elif v is None:
                v = ""
            elif not isinstance(v, (str, int, float, bool)):
                v = str(v)

            row[k] = v
        norm_rows.append(row)

    results_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(norm_rows, columns=cols)
    df.to_csv(results_csv, index=False, encoding="utf-8")


def is_row_done(
    row: pd.Series,
    output_dir: Path,
    mseed_dirname: str,
) -> bool:
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
    net = str(row["net"]).strip()
    sta = str(row["sta"]).strip()
    loc_field = str(row["loc"]).strip()
    chan_prefix = str(row["chan"]).strip()

    record_sub = f"{net}_{sta}_{chan_prefix}_{loc_field}"
    mseed_path = output_dir / mseed_dirname / net / record_sub

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

            chan_check = parts[3].split("__", 1)[0]
            if not chan_check.endswith("Z"):
                continue

            if "__" not in stem:
                continue

            last = stem.rsplit("__", 1)[-1]
            if last == target_end:
                return True
    except OSError:
        return False

    return False


def create_output_dirs(
    net: str,
    sta: str,
    chan_prefix: str,
    loc: str,
    output_dir: Path,
    mseed_dirname: str,
    stationxml_dirname: str,
) -> tuple[Path, Path]:
    """Create output directories for a network/station/channel/location.

    Parameters
    ----------
    net : str
        Network code.
    sta : str
        Station code.
    chan_prefix : str
        2-character channel prefix (e.g. ``"HH"``).
    loc : str
        Location code used for naming the output directory.
    output_dir : Path
        Root output directory.
    mseed_dirname : str
        MiniSEED subdirectory name.
    stationxml_dirname : str
        StationXML subdirectory name.

    Returns
    -------
    tuple[Path, Path]
        Tuple of ``(mseed_path, xml_path)``.
    """
    record_sub = f"{net}_{sta}_{chan_prefix}_{loc}"

    mseed_path = output_dir / mseed_dirname / net / record_sub
    xml_path = output_dir / stationxml_dirname / net / record_sub

    mseed_path.mkdir(parents=True, exist_ok=True)
    xml_path.mkdir(parents=True, exist_ok=True)

    return mseed_path, xml_path


def download_task(
    task: tuple[int, str, dict[str, object]],
    *,
    output_dir: Path,
    mseed_dirname: str,
    stationxml_dirname: str,
    month_seconds: int,
    min_chunk_seconds: int,
) -> dict[str, int | str] | None:
    """Download waveform and StationXML data for a single CSV row.

    Parameters
    ----------
    task : tuple[int, str, dict[str, object]]
        A 3-tuple of ``(idx, provider, row_dict)``.
    output_dir : Path
        Root output directory.
    mseed_dirname : str
        MiniSEED subdirectory name.
    stationxml_dirname : str
        StationXML subdirectory name.
    month_seconds : int
        Default chunk length (seconds) for large windows.
    min_chunk_seconds : int
        Minimum chunk length (seconds) when backing off after 413 errors.

    Returns
    -------
    dict[str, int | str] | None
        A result dictionary with keys including ``idx``, ``status``, ``provider``,
        ``net``, ``sta``, and optionally ``error``.

    Raises
    ------
    Exception
        Re-raises exceptions encountered during download attempts when the
        request cannot be reduced any further.
    """
    idx, provider, row = task

    try:
        net = str(row["net"]).strip()
        sta = str(row["sta"]).strip()
        loc_field = str(row["loc"]).strip()
        loc = "*" if loc_field == "NA" else loc_field
        chan_prefix = str(row["chan"]).strip()
        channel = f"{chan_prefix}?"

        start = UTCDateTime(row["start_date"])
        end = UTCDateTime(row["end_date"])

        total_window = end - start
        chunk_base = int(min(total_window, month_seconds))

        max_attempts = 4
        attempt = 1

        print(
            f"[{idx}] Provider={provider} Downloading {net}.{sta} {channel} {start} -> {end} chunk={chunk_base}s"
        )

        mseed_path, xml_path = create_output_dirs(
            net,
            sta,
            chan_prefix,
            loc_field,
            output_dir=output_dir,
            mseed_dirname=mseed_dirname,
            stationxml_dirname=stationxml_dirname,
        )

        while attempt <= max_attempts:
            chunklength = int(min(total_window, chunk_base))
            if chunklength < 1:
                chunklength = 1

            domain = GlobalDomain()
            restrictions = Restrictions(
                starttime=start,
                endtime=end,
                chunklength_in_sec=chunklength,
                network=net,
                station=sta,
                location=loc,
                channel=channel,
                reject_channels_with_gaps=False,
                minimum_length=0.0,
                minimum_interstation_distance_in_m=0.0,
            )

            try:
                mdl = MassDownloader(providers=[provider])
                mdl.download(
                    domain,
                    restrictions,
                    mseed_storage=str(mseed_path),
                    stationxml_storage=str(xml_path),
                )

                print(f"[{idx}] Done")
                return {
                    "idx": idx,
                    "status": "ok",
                    "provider": provider,
                    "net": net,
                    "sta": sta,
                }
            except Exception as exc:  # noqa: BLE001
                err_text = f"{exc!r} {exc}"
                is_manifest_too_large = (
                    "Estimated manifest size" in err_text
                    or "Request Entity Too Large" in err_text
                    or "413" in err_text
                )

                if is_manifest_too_large:
                    if chunk_base <= min_chunk_seconds:
                        print(
                            f"[{idx}] Server denied request and chunk is already at minimum ({chunk_base}s). Giving up."
                        )
                        raise

                    old = chunk_base
                    chunk_base = max(min_chunk_seconds, chunk_base // 2)
                    print(
                        f"[{idx}] Server denied request (413). Reducing chunk base {old}s -> {chunk_base}s and retrying."
                    )
                    attempt += 1
                    continue

                raise

    except Exception as exc:  # noqa: BLE001
        print(
            f"[{idx}] ERROR provider={provider} net={row.get('net')} sta={row.get('sta')}: {exc}"
        )
        return {
            "idx": idx,
            "status": "error",
            "error": str(exc),
            "provider": provider,
        }


@app.command()
def run(
    csv_file: Annotated[
        Path,
        typer.Argument(
            ..., exists=True, dir_okay=False, help="Input station/channel CSV."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            ...,
            file_okay=False,
            help="Root output directory for waveforms and StationXML.",
        ),
    ],
    results_csv: Annotated[
        Path | None,
        typer.Option(
            dir_okay=False,
            help="Optional output CSV for per-row download results (default: <output_dir>/download_results.csv).",
        ),
    ] = None,
    mseed_dirname: Annotated[
        str,
        typer.Option(help="MiniSEED subdirectory name."),
    ] = "waveforms",
    stationxml_dirname: Annotated[
        str,
        typer.Option(help="StationXML subdirectory name."),
    ] = "stationxml",
) -> None:
    """Download waveforms + StationXML for each row in a station/channel request table.

    Parameters
    ----------
    csv_file : Path
        Input CSV describing requests. Must contain columns:
        ``net``, ``sta``, ``loc``, ``chan``, ``start_date``, ``end_date``, ``provider``.
    output_dir : Path
        Root output directory.
    results_csv : Path | None, optional
        Output CSV for results. If omitted, defaults to
        ``<output_dir>/download_results.csv``.
    mseed_dirname : str, optional
        MiniSEED subdirectory name.
    stationxml_dirname : str, optional
        StationXML subdirectory name.

    Returns
    -------
    None
        This function returns ``None``.

    Raises
    ------
    ValueError
        If required columns are missing from the input CSV.
    """
    df = pd.read_csv(csv_file, dtype={"loc": str}, keep_default_na=False)

    required_cols = {"net", "sta", "loc", "chan", "start_date", "end_date", "provider"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"CSV must contain: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if results_csv is None:
        results_csv = output_dir / "download_results.csv"

    tasks: list[tuple[int, str, dict[str, object]]] = []
    skipped = 0
    for idx, row in df.reset_index(drop=True).iterrows():
        if is_row_done(row, output_dir=output_dir, mseed_dirname=mseed_dirname):
            skipped += 1
            continue

        tasks.append((int(idx), str(row["provider"]), row.to_dict()))

    print(
        f"Starting sequential run for {len(tasks)} tasks (skipped {skipped} already done)"
    )

    results: list[dict[str, object]] = []
    for task in tasks:
        results.append(
            download_task(
                task,
                output_dir=output_dir,
                mseed_dirname=mseed_dirname,
                stationxml_dirname=stationxml_dirname,
                month_seconds=MONTH_SECONDS,
                min_chunk_seconds=MIN_CHUNK_SECONDS,
            )
        )

    save_results(results, results_csv=results_csv)

    oks = sum(1 for r in results if r.get("status") == "ok")
    errs = sum(1 for r in results if r.get("status") == "error")
    print(
        f"Completed: {oks} succeeded, {errs} failed (skipped {skipped} previously done)"
    )


if __name__ == "__main__":
    app()
