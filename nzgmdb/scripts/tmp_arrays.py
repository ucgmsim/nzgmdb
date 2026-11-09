"""
Commands to find, bulk download, check and back up temporary array data.

This runs separately from the main pipeline. The pipeline then reads the
downloaded waveforms with `--tmp-array-dir <data_dir>/waveforms`.
See wiki/Temporary-Arrays.md for the full workflow.
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.tmp_array import backup, completeness, download, file_layout, stations
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)


def _split_commas(values: list[str] | None) -> list[str]:
    """
    Split a comma separated typer option into a list.

    Parameters
    ----------
    values : list[str] | None
        Raw option values

    Returns
    -------
    list[str]
        Split values, empty if the option was not given
    """
    return [v for x in values or [] for v in x.split(",") if v]


def _window_name(start: datetime, end: datetime) -> str:
    """
    Get a short name for a time window used in default output filenames.

    Parameters
    ----------
    start : datetime
        Window start
    end : datetime
        Window end

    Returns
    -------
    str
        Name like 20260101_20270101
    """
    return f"{start:%Y%m%d}_{end:%Y%m%d}"


@cli.from_docstring(app)
def get_stations(
    station_ffp: Annotated[Path, typer.Argument(dir_okay=False)],
    start_date: Annotated[datetime, typer.Argument()],
    end_date: Annotated[datetime, typer.Argument()],
    providers: Annotated[
        list[str], typer.Option(callback=_split_commas)
    ] = stations.DEFAULT_PROVIDERS,
    channels: Annotated[
        list[str], typer.Option(callback=_split_commas)
    ] = stations.DEFAULT_CHANNELS,
    exclude_networks: Annotated[
        list[str], typer.Option(callback=_split_commas)
    ] = stations.DEFAULT_EXCLUDED_NETWORKS,
    coastline_filter: Annotated[bool, typer.Option()] = True,
    min_lat: Annotated[float, typer.Option()] = -49.0,
    max_lat: Annotated[float, typer.Option()] = -32.0,
    min_lon: Annotated[float, typer.Option()] = 165.0,
    max_lon: Annotated[float, typer.Option()] = -176.9,
):
    """
    Find temporary array stations in NZ operating between the dates and write the station CSV.

    Prints any networks found that are not yet in config.yaml, as the pipeline only
    fetches inventories for networks listed there.

    Parameters
    ----------
    station_ffp : Path
        Output station CSV
    start_date : datetime
        Only stations operating after this date
    end_date : datetime
        Only stations operating before this date
    providers : list[str], optional
        Comma separated FDSN providers, earlier ones win for duplicate stations
    channels : list[str], optional
        Comma separated 2 character channel prefixes to keep
    exclude_networks : list[str], optional
        Comma separated networks to drop (NZ and IU come from GeoNet in the main pipeline)
    coastline_filter : bool, optional
        Keep only stations inside the NZ coastline
    min_lat : float, optional
        Minimum latitude
    max_lat : float, optional
        Maximum latitude
    min_lon : float, optional
        Minimum longitude
    max_lon : float, optional
        Maximum longitude, less than min_lon wraps across 180
    """
    station_df = stations.get_station_table(
        file_layout.to_utc(start_date),
        file_layout.to_utc(end_date),
        providers,
        channels,
        exclude_networks,
        (min_lat, max_lat, min_lon, max_lon),
        coastline_filter,
    )
    station_ffp.parent.mkdir(parents=True, exist_ok=True)
    station_df.to_csv(station_ffp, index=False)
    print(f"Wrote {len(station_df)} station rows to {station_ffp}")

    missing_df = stations.find_networks_missing_from_config(station_df)
    if missing_df.empty:
        print("All networks found are already in config.yaml")
        return
    print(
        "\nThese networks are not in config.yaml (tmp_array_providers_networks), "
        "so the pipeline will not fetch their inventories. Add them if they should be used:"
    )
    print(missing_df.to_string(index=False))


@cli.from_docstring(app)
def download_data(
    station_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    data_dir: Annotated[Path, typer.Argument(file_okay=False)],
    start_date: Annotated[datetime, typer.Argument()],
    end_date: Annotated[datetime, typer.Argument()],
    results_ffp: Annotated[Path, typer.Option(dir_okay=False)] = None,
):
    """
    Download continuous waveforms and StationXML for every station between the dates.

    Adds to any existing data in data_dir and skips months already downloaded, so it
    is safe to re-run or to run again later with a new window. Use dates on month
    boundaries (e.g. 2026-01-01 to 2027-01-01) so windows do not overlap.

    Parameters
    ----------
    station_ffp : Path
        Station CSV from get-stations
    data_dir : Path
        Root temporary array data directory
    start_date : datetime
        Window start
    end_date : datetime
        Window end, also used for stations that are still running
    results_ffp : Path, optional
        Output CSV of per row results, by default <data_dir>/download_results_<start>_<end>.csv
    """
    station_df = file_layout.read_station_csv(station_ffp)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_df = download.download_stations(
        station_df,
        data_dir,
        file_layout.to_utc(start_date),
        file_layout.to_utc(end_date),
    )

    if results_ffp is None:
        results_ffp = (
            data_dir / f"download_results_{_window_name(start_date, end_date)}.csv"
        )
    results_df.to_csv(results_ffp, index=False)
    print(f"\n{results_df['status'].value_counts().to_string()}")
    print(f"Wrote results to {results_ffp}")


@cli.from_docstring(app)
def check_data(
    station_ffp: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    data_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    start_date: Annotated[datetime, typer.Argument()],
    end_date: Annotated[datetime, typer.Argument()],
    output_ffp: Annotated[Path, typer.Option(dir_okay=False)] = None,
    providers: Annotated[list[str], typer.Option(callback=_split_commas)] = None,
):
    """
    Report which stations are complete, partial or missing between the dates.

    Parameters
    ----------
    station_ffp : Path
        Station CSV from get-stations
    data_dir : Path
        Root temporary array data directory
    start_date : datetime
        Window start
    end_date : datetime
        Window end
    output_ffp : Path, optional
        Output CSV, by default <data_dir>/check_<start>_<end>.csv
    providers : list[str], optional
        Comma separated providers to check, by default all
    """
    station_df = file_layout.read_station_csv(station_ffp)
    if providers:
        station_df = station_df[station_df["provider"].isin(providers)]
    check_df = completeness.check_stations(
        station_df,
        data_dir,
        file_layout.to_utc(start_date),
        file_layout.to_utc(end_date),
    )

    if output_ffp is None:
        output_ffp = data_dir / f"check_{_window_name(start_date, end_date)}.csv"
    check_df.to_csv(output_ffp, index=False)
    print(f"\nWrote check results to {output_ffp}")


@cli.from_docstring(app)
def backup_data(
    data_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    dropbox_path: Annotated[str, typer.Argument()],
):
    """
    Back up the data directory to Dropbox with rclone, uploading only new or changed folders.

    Parameters
    ----------
    data_dir : Path
        Root temporary array data directory
    dropbox_path : str
        rclone Dropbox path to upload to
    """
    manifest_df = backup.backup_to_dropbox(data_dir, dropbox_path)
    print(f"\n{manifest_df['status'].value_counts().to_string()}")


if __name__ == "__main__":
    app()
