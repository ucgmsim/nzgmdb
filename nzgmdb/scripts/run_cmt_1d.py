"""
Script to run the near-real-time 1-D CMT inversion (BayesISOLA) for a
specific event.
"""

import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from obspy.clients.fdsn import Client as FDSN_Client

from BayesISOLA.gf_helpers import build_regular_velocity_grid
from BayesISOLA.workflows import get_mseed_stationxml, run_auto_cmt
from qcore import cli

app = typer.Typer(pretty_exceptions_enable=False)

# 3-D NZ velocity model used to build station/path-specific 1-D Axitra
# models. Override with --nz-3dvm-path if a different local copy is used.
DEFAULT_NZ_3DVM_PATH = Path("/scratch/SeismicNow/data/cmt_1d_data/nz3dvm_2p3.csv")
DEFAULT_THREADS = 8  # tune to the number of cores available on the machine this runs on


@cli.from_docstring(app)
def run_cmt_1d(
    event_id: Annotated[str, typer.Argument()],
    event_csv_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
        ),
    ],
    nz_3dvm_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
        ),
    ] = DEFAULT_NZ_3DVM_PATH,
    threads: Annotated[int, typer.Option()] = DEFAULT_THREADS,
    min_radius_km: Annotated[float, typer.Option()] = 0.0,
    max_radius_km: Annotated[float, typer.Option()] = None,
) -> dict:
    """
    Run the near-real-time 1-D CMT inversion for one event.

    The production batch pipeline calls
    ``run_auto_cmt(..., waveform_source="fdsn", client="GEONET")``, which
    lets BayesISOLA reach out to FDSN and download waveforms itself.
    GeoNet's standard FDSN service ("GEONET") is not updated in near real
    time, so a recent event is not there yet. This command works around
    that by doing the acquisition step manually:

    1. Build an obspy FDSN client pointed at GeoNet's near-real-time
       service (``service-nrt.geonet.org.nz``), tried first so a
       just-happened event is picked up before it reaches the standard
       archive.
    2. Call ``BayesISOLA.workflows.get_mseed_stationxml()`` directly with
       that client and the standard "GEONET" client as a fallback. This is
       the exact same station-discovery/download routine that
       ``run_auto_cmt(waveform_source="fdsn")`` calls internally.
       ``get_mseed_stationxml`` discovers candidates from every client in
       the list and, if a station's download fails under an earlier client
       (e.g. too old for the near-real-time service's short rolling
       buffer), retries it under the next one automatically. It downloads
       miniSEED + StationXML under ``<output_dir>/raw`` and writes station
       metadata under ``<output_dir>/metadata``, returning a station table
       with local file paths.
    3. Pass that station table straight into
       ``run_auto_cmt(waveform_source="local", station_df=...)``, which
       skips acquisition entirely and inverts the files already on disk.

    Parameters
    ----------
    event_id : str
        GeoNet event/public ID, e.g. "2026p576643".
    event_csv_path : Path
        CSV with the event row (evid, datetime, lat, lon, depth, mag, ...) -
        same schema as GeoNet's earthquake_source_table.csv.
    output_dir : Path
        Directory raw/, metadata/, input/, results/, figures/ get written
        under.
    nz_3dvm_path : Path, optional
        3-D NZ velocity model CSV.
    threads : int, optional
        Axitra thread count.
    min_radius_km : float, optional
        Inner radius (km) of the station search annulus.
    max_radius_km : float, optional
        Outer radius (km) of the station search annulus. Omit to resolve it
        automatically from the event magnitude.

    Returns
    -------
    dict
        The ``run_auto_cmt`` return value; ``run["results"]`` holds the
        curated centroid/summary/station-fit tables.
    """
    start_time = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw"
    metadata_path = output_dir / "metadata"
    input_path = output_dir / "input"
    input_path.mkdir(parents=True, exist_ok=True)

    print("Event directory :", output_dir)
    print("Raw data        :", raw_path)
    print("Metadata        :", metadata_path)
    print("BayesISOLA input:", input_path)

    # -----------------------------------------------------------------
    # Event parameters
    # -----------------------------------------------------------------

    event_df = pd.read_csv(event_csv_path)

    if "evid" in event_df.columns and str(event_df["evid"].values[0]) != event_id:
        print(f"WARNING: event_id={event_id!r} does not match evid="
              f"{event_df['evid'].values[0]!r} in {event_csv_path}")

    event_time = event_df["datetime"].values[0]
    lon_event = event_df["lon"].values[0]
    lat_event = event_df["lat"].values[0]
    depth_km = event_df["depth"].values[0]
    mag_event = event_df["mag"].values[0]

    print(f"Event {event_id}: t={event_time}  lon={lon_event}  lat={lat_event}  "
          f"depth={depth_km} km  mag={mag_event}")

    # -----------------------------------------------------------------
    # Velocity model
    # -----------------------------------------------------------------

    nz_3dvm = pd.read_csv(nz_3dvm_path)

    nz_grid_ll = build_regular_velocity_grid(
        nz_3dvm,
        x_col="Longitude",
        y_col="Latitude",
        depth_col="Depth(km_BSL)",
        vp_col="Vp",
        vs_col="Vs",
        density_col="Density",
        qs_col="Qs",
        qp_col="Qp",
        coordinate_crs="EPSG:4326",
        interpolation_crs="EPSG:2193",
    )

    # -----------------------------------------------------------------
    # Step 1: manually discover stations + download waveforms/StationXML.
    # -----------------------------------------------------------------

    client_NZ = FDSN_Client(base_url="https://service-nrt.geonet.org.nz")
    client = [client_NZ, "GEONET"]

    print("\nDownloading waveforms + StationXML (NRT client, falling back to GEONET)...")

    nrt_event_df, station_df, download_log, waveform_window = get_mseed_stationxml(
        event_id, event_time, lon_event, lat_event, depth_km,
        magnitude=mag_event,
        output_dir=output_dir,
        client=client,
        min_radius_km=min_radius_km,
        max_radius_km=max_radius_km,
        ground_level=True,
        channels=("HH?", "BH?", "LH?"),
        channel_priority=("HH", "BH", "LH"),
        time_unc_s=2.0,
        min_depth_km=5.0,
        min_depth_multiplier=0.5,
        max_depth_multiplier=3.0,
        rupture_velocity_m_s=1000.0,
        velocity_slowest_m_s=1000.0,
        covariance="noise",
        overwrite=False,
        plot=True,
        show=False,
    )

    print(f"Downloaded {len(station_df)} station(s):")
    print(station_df[["station_id", "distance_km", "download_status"]].to_string(index=False))

    if not download_log.empty:
        failed = download_log[download_log["status"].isin(["download_failed", "client_failed"])]
        if not failed.empty:
            print(f"\n{len(failed)} station(s) failed to download - see "
                  f"{metadata_path / 'download_log.csv'} for details.")

    # -----------------------------------------------------------------
    # Step 2: run the inversion against the local files just downloaded.
    # -----------------------------------------------------------------

    print("\nRunning CMT inversion against local waveform data...")

    run = run_auto_cmt(
        event_id, event_time, lon_event, lat_event, depth_km, mag_event,
        output_dir=output_dir,
        velocity_model=nz_grid_ll,
        gf_source="axitra",
        waveform_source="local",
        station_df=station_df,
        min_radius_km=min_radius_km,
        max_radius_km=max_radius_km,
        ground_level=True,
        channels=("HH?", "BH?", "LH?"),
        channel_priority=("HH", "BH", "LH"),
        location_unc_km=0.0,
        time_unc_s=2.0,
        min_depth_km=5.0,
        min_depth_multiplier=0.5,
        max_depth_multiplier=3.0,
        step_x_km=2.0,
        step_z_km=1.0,
        max_grid_points=5000,
        add_rupture_length=True,
        rupture_velocity_m_s=1000.0,
        velocity_slowest_m_s=1000.0,
        freqmin=0.02,
        freqmax=0.05,
        threads=threads,
        use_precalculated_Green="auto",
        covariance="noise",
        crosscovariance=True,
        n_uncertainty=None,
        plot=True,
        plot_preset="summary",
        show=False,
    )

    print("Complete")
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")

    results = run["results"]
    print("\nCentroid:")
    print(results["centroid"])
    print("\nSummary:")
    print(results["summary"])

    return run


if __name__ == "__main__":
    app()
