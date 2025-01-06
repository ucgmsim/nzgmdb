import datetime
import functools
import io
import multiprocessing as mp
import shutil
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import requests
import scipy as sp
import typer
from obspy.clients.fdsn import Client as FDSN_Client

from nzgmdb.data_retrieval import geonet, sites
from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure
from nzgmdb.scripts import run_nzgmdb

app = typer.Typer()

SEISMIC_NOW_URL = (
    "https://quakecoresoft.canterbury.ac.nz/seismicnow/api/earthquakes/add"
)


def download_earthquake_data(
    start_date: datetime, end_date: datetime, mag_filter: float
) -> pd.DataFrame:
    """
    Download the earthquake data from the GeoNet API

    Parameters
    ----------
    start_date : datetime
        Start date to download the data
    end_date : datetime
        End date to download the data
    mag_filter : float
        Minimum magnitude to filter the data

    Returns
    -------
    pd.DataFrame
        Dataframe containing the earthquake data
    """
    # Format the dates in the required format
    start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
    end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")

    # Define bbox for New Zealand
    config = cfg.Config()
    bbox = ",".join([str(coord) for coord in config.get_value("bbox")])

    # Send API request for the last hour
    geonet_url = config.get_value("geonet_url")
    endpoint = f"{geonet_url}/csv?bbox={bbox}&startdate={start_date_str}&enddate={end_date_str}"
    response = requests.get(endpoint)

    # Check if the response is valid
    if response.status_code != 200:
        raise ValueError("Could not get the earthquake data")

    # Read the response into a dataframe
    df = pd.read_csv(io.StringIO(response.text))

    # Filter by magnitude
    df = df[df["magnitude"] >= mag_filter]

    return df


def custom_multiprocess_geonet(event_dir: Path, event_id: str, n_procs: int = 1):
    """
    Multiprocess the GeoNet data retrieval for speed efficiency over stations

    Parameters
    ----------
    event_dir : Path
        The directory for the event
    event_id : str
        The event ID
    n_procs : int
        The number of processes to use

    Returns
    -------
    bool
        True if the event was processed, False if the event was skipped
    """
    # Generate the site basin flatfile
    flatfile_dir = file_structure.get_flatfile_dir(event_dir)
    flatfile_dir.mkdir(parents=True, exist_ok=True)

    site_df = sites.create_site_table_response()
    site_df = sites.add_site_basins(site_df)

    site_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.SITE_TABLE, index=False
    )

    # Set constants
    config = cfg.Config()
    channel_codes = ",".join(config.get_value("channel_codes"))
    client_NZ = FDSN_Client(base_url=config.get_value("real_time_url"))
    inventory = client_NZ.get_stations(channel=channel_codes, level="response")
    # Get the catalog information
    cat = client_NZ.get_events(eventid=event_id)
    event_cat = cat[0]

    # Get the event line and save the output
    event_line = geonet.fetch_event_line(event_cat, event_id)
    event_df = pd.DataFrame(
        [event_line],
        columns=[
            "evid",
            "datetime",
            "lat",
            "lon",
            "depth",
            "loc_type",
            "loc_grid",
            "mag",
            "mag_type",
            "mag_method",
            "mag_unc",
            "mag_orig",
            "mag_orig_type",
            "mag_orig_unc",
            "ndef",
            "nsta",
            "nmag",
            "t_res",
            "reloc",
        ],
    )
    # Save the dataframes with a suffix
    event_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_GEONET,
        index=False,
    )

    # Get the data_dir
    data_dir = file_structure.get_data_dir()

    mw_rrup = np.loadtxt(data_dir / "Mw_rrup.txt")
    mags = mw_rrup[:, 0]
    rrups = mw_rrup[:, 1]
    # Generate cubic interpolation for magnitude distance relationship
    f_rrup = sp.interpolate.interp1d(mags, rrups, kind="cubic")

    # Get Networks / Stations within a certain radius of the event
    inv_sub_sta = geonet.get_stations_within_radius(
        event_cat, mags, rrups, f_rrup, inventory
    )

    # Check if there are any stations
    if len(inv_sub_sta) == 0:
        return False

    sta_mag_table = []
    skipped_records = []

    # Iterate over the networks
    for network in inv_sub_sta:
        # Generate the stations to process
        stations = [station for station in network]

        # Fetch results
        with mp.Pool(n_procs) as p:
            results = p.map(
                functools.partial(
                    geonet.fetch_sta_mag_line,
                    network=network,
                    event_cat=event_cat,
                    event_id=event_id,
                    event_dir=event_dir,
                    client_NZ=client_NZ,
                    pref_mag=event_line[8],
                    pref_mag_type=event_line[9],
                    site_table=site_df,
                ),
                stations,
            )

        # Extract the results
        for result in results:
            finished_sta_mag_table, finished_skipped_records = result
            sta_mag_table.extend(finished_sta_mag_table)
            skipped_records.extend(finished_skipped_records)

    if len(sta_mag_table) == 0:
        # No station data, skip this event
        return False
    else:
        sta_mag_df = pd.DataFrame(
            sta_mag_table,
            columns=[
                "magid",
                "net",
                "sta",
                "loc",
                "chan",
                "evid",
                "mag",
                "mag_type",
                "mag_corr_method",
                "amp",
                "amp_unit",
            ],
        )

    sta_mag_df.to_csv(
        flatfile_dir / file_structure.PreFlatfileNames.STATION_MAGNITUDE_TABLE_GEONET,
        index=False,
    )

    if len(skipped_records) > 0:
        # Create the skipped records df
        skipped_records_df = pd.DataFrame(
            skipped_records, columns=["skipped_records", "reason"]
        )
    else:
        skipped_records_df = pd.DataFrame()

    skipped_records_df.to_csv(
        flatfile_dir / file_structure.SkippedRecordFilenames.GEONET_SKIPPED_RECORDS,
        index=False,
    )
    return True


@app.command(
    help="Run the NZGMDB pipeline for a specific event in near-real-time mode."
)
def run_event(  # noqa: D103
    event_id: Annotated[
        str,
        typer.Argument(
            help="The event ID.",
        ),
    ],
    event_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory for the event to output to.",
            exists=True,
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
    conda_sh: Annotated[
        Path,
        typer.Argument(
            help="Path to the conda.sh script for environment activation.",
            exists=True,
            file_okay=True,
        ),
    ],
    gmc_activate: Annotated[
        str,
        typer.Argument(
            help="Command to activate gmc environment.",
        ),
    ],
    gmc_predict_activate: Annotated[
        str,
        typer.Argument(
            help="Command to activate gmc_predict environment.",
        ),
    ],
    n_procs: Annotated[
        int,
        typer.Option(
            help="Number of processes to use in the pipeline.",
        ),
    ] = 1,
    gmc_procs: Annotated[
        int,
        typer.Option(
            help="Number of GMC processes to use.",
        ),
    ] = 1,
    ko_matrix_path: Annotated[
        Path,
        typer.Option(
            help="Path to the KO matrix directory.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    add_seismic_now: Annotated[
        bool,
        typer.Option(
            help="Add the event to SeismicNow.",
            is_flag=True,
        ),
    ] = False,
    start_date: Annotated[
        datetime.datetime,
        typer.Option(
            help="Start date for the event.",
        ),
    ] = datetime.datetime.utcnow()
    - datetime.timedelta(days=8),
    end_date: Annotated[
        datetime.datetime,
        typer.Option(
            help="End date for the event.",
        ),
    ] = datetime.datetime.utcnow()
    - datetime.timedelta(minutes=1),
):
    """
    Run the NZGMDB pipeline for a specific event in near-real-time mode.

    Returns
    -------
    bool
        True if the event was processed, False if the event was skipped
    """

    # Run the custom multiprocess geonet, site table and geonet steps
    result = custom_multiprocess_geonet(event_dir, event_id, n_procs)

    if result is None:
        # Skip this event
        print(f"Skipping event {event_id}")
        return None

    # Run the rest of the pipeline
    run_nzgmdb.run_full_nzgmdb(
        event_dir,
        start_date,
        end_date,
        gm_classifier_dir,
        conda_sh,
        gmc_activate,
        gmc_predict_activate,
        gmc_procs,
        n_procs,
        ko_matrix_path=ko_matrix_path,
        checkpoint=True,
        only_event_ids=[event_id],
        real_time=True,
    )

    if add_seismic_now:
        # Define the URL for the endpoint
        url = f"{SEISMIC_NOW_URL}?earthquake_id={event_id}"

        # Send a POST request to the endpoint
        response = requests.post(url)

        # Check the response status
        if response.status_code == 200:
            print("Event added successfully")
        else:
            print(f"Failed to add event. Status code: {response.status_code}")
            print(f"Response: {response.text}")
    return True


@app.command(
    help="Poll earthquake data, process events, and run the NZGMDB pipeline for real-time data."
)
def poll_earthquake_data(  # noqa: D103
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory for earthquake event data.",
            exists=True,
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
    conda_sh: Annotated[
        Path,
        typer.Argument(
            help="Path to the conda.sh script for environment activation.",
            exists=True,
            file_okay=True,
        ),
    ],
    gmc_activate: Annotated[
        str,
        typer.Argument(
            help="Command to activate gmc environment.",
        ),
    ],
    gmc_predict_activate: Annotated[
        str,
        typer.Argument(
            help="Command to activate gmc_predict environment.",
        ),
    ],
    n_procs: Annotated[
        int,
        typer.Option(
            help="Number of processes to use in the pipeline.",
        ),
    ] = 1,
    gmc_procs: Annotated[
        int,
        typer.Option(
            help="Number of GMC processes to use.",
        ),
    ] = 1,
    ko_matrix_path: Annotated[
        Path,
        typer.Option(
            help="Path to the KO matrix directory.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    add_seismic_now: Annotated[
        bool,
        typer.Option(
            help="Add the event to SeismicNow.",
            is_flag=True,
        ),
    ] = False,
):
    init_start_date = None
    while True:
        # Get the last 2 minutes worth of data and check if there are any new events
        end_date = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
        # If an event was just executed, ensures we capture any events that may have been missed during the execution
        start_date = (
            end_date - datetime.timedelta(minutes=2)
            if init_start_date is None
            else init_start_date
        )
        geonet_df = download_earthquake_data(start_date, end_date, mag_filter=4.0)
        init_start_date = None

        if not geonet_df.empty:
            # Run every event
            for event_id in geonet_df["publicid"].values:
                event_dir = main_dir / str(event_id)
                # If the event exists skip
                if event_dir.exists():
                    print(f"Event {event_id} already exists")
                    continue
                event_dir.mkdir()

                result = run_event(
                    str(event_id),
                    event_dir,
                    gm_classifier_dir,
                    conda_sh,
                    gmc_activate,
                    gmc_predict_activate,
                    n_procs,
                    gmc_procs,
                    ko_matrix_path,
                    add_seismic_now,
                    start_date,
                    end_date,
                )

                if result is None:
                    # remove the event directory
                    shutil.rmtree(event_dir)
                init_start_date = end_date

        time.sleep(60)


if __name__ == "__main__":
    app()
