"""
Script to run the NZGMDB pipeline for a specific event in near-real-time mode.
"""

import datetime
import io
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
import requests
import typer

from IM import ims
from nzgmdb.management import config as cfg
from nzgmdb.management import custom_errors, file_structure
from nzgmdb.scripts import run_gmc, run_nzgmdb
from qcore import cli

app = typer.Typer()

SEISMIC_NOW_URL = (
    "https://quakecoresoft.canterbury.ac.nz/seismicnow/api/earthquakes/add"
)
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL")


def send_message_to_slack(message: str):
    """
    Send a message to a Slack channel.

    Parameters:
    ----------
    message (str): The message to send to the Slack channel.

    Returns:
    -------
        dict: The response JSON containing the message timestamp (ts)
    """
    if not SLACK_CHANNEL:
        raise ValueError(
            "No slack channel provided from the environment var SLACK_CHANNEL"
        )
    if not SLACK_BOT_TOKEN:
        raise ValueError(
            "No slack bot token provided from the environment var SLACK_BOT_TOKEN"
        )
    url = "https://slack.com/api/chat.postMessage"
    data = {
        "channel": SLACK_CHANNEL,
        "text": message,
    }
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()

    if not response_data.get("ok"):
        raise ValueError(f"Error sending message: {response_data}")

    return response_data


def reply_to_message_on_slack(thread_ts: str, reply_message: str):
    """
    Reply to a message in Slack (threaded reply).

    Parameters:
    ----------
    thread_ts (str):
        The timestamp of the message to reply to.
    reply_message (str):
        The reply text.

    Returns:
    -------
        dict: The response JSON containing the message timestamp (ts)
    """
    url = "https://slack.com/api/chat.postMessage"
    data = {
        "channel": SLACK_CHANNEL,
        "text": reply_message,
        "thread_ts": thread_ts,  # This ensures it's a threaded reply
    }
    HEADERS = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=HEADERS, json=data)
    response_data = response.json()

    if not response_data.get("ok"):
        raise ValueError(f"Error sending reply: {response_data}")

    return response_data


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


def update_eq_source_table(
    event_dir: Path,
):
    """
    Update the earthquake source table with the latest data

    Parameters
    ----------
    event_dir : Path
        The event directory
    """
    flatfile_dir = file_structure.get_flatfile_dir(event_dir)
    eq_source_ffp = (
        flatfile_dir / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_GEONET
    )
    eq_source_df = pd.read_csv(eq_source_ffp)

    # Get the datetime of the event
    datetime_evid = eq_source_df["datetime"].values[0]
    datetime_evid = pd.to_datetime(datetime_evid)

    # Get the latest data
    geonet_df = download_earthquake_data(
        datetime_evid - timedelta(minutes=2),
        datetime_evid + timedelta(minutes=2),
        mag_filter=0.0,
    )

    # Get the new data for this event
    new_data = geonet_df[geonet_df["publicid"] == eq_source_df["evid"].values[0]]

    # Update the data
    eq_source_df["mag"] = new_data["magnitude"].values[0]
    eq_source_df["lat"] = new_data["latitude"].values[0]
    eq_source_df["lon"] = new_data["longitude"].values[0]
    eq_source_df["depth"] = new_data["depth"].values[0]

    # Save the updated data
    eq_source_df.to_csv(eq_source_ffp, index=False)

    # return the new data values
    return (
        eq_source_df["mag"].values[0],
        eq_source_df["lat"].values[0],
        eq_source_df["lon"].values[0],
        eq_source_df["depth"].values[0],
    )


@cli.from_docstring(app)
def run_event(
    event_id: Annotated[str, typer.Argument()],
    event_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    gm_classifier_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    conda_sh: Annotated[Path, typer.Argument(exists=True, file_okay=True)],
    gmc_activate: Annotated[str, typer.Argument()],
    gmc_predict_activate: Annotated[str, typer.Argument()],
    ko_matrix_path: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    add_seismic_now: Annotated[bool, typer.Option(is_flag=True)] = False,
    machine: Annotated[
        cfg.MachineName,
        typer.Option(
            case_sensitive=False,
        ),
    ] = cfg.MachineName.LOCAL,
):
    """
    Run the NZGMDB pipeline for a specific event in near-real-time mode.

    Parameters
    ----------
    event_id : str
        The event ID.
    event_dir : Path
        The directory for the event to output to.
    gm_classifier_dir : Path
        Directory for gm_classifier.
    conda_sh : Path
        Path to the conda.sh script for environment activation.
    gmc_activate : str
        Command to activate gmc environment.
    gmc_predict_activate : str
        Command to activate gmc_predict environment.
    ko_matrix_path : Path, optional
        Path to the KO matrix directory (default is None).
    add_seismic_now : bool, optional
        Whether to add the event to SeismicNow (default is False).
    machine : cfg.MachineName, optional
        The machine name to use for the number of processes (default is cfg.MachineName.LOCAL).

    Returns
    -------
    bool
        True if the event was processed, False if the event was skipped.
    """
    try:
        config = cfg.Config()
        # Execute custom pipeline
        run_nzgmdb.generate_site_table_basin(event_dir)
        # Get geonet data
        run_nzgmdb.fetch_geonet_data(
            event_dir,
            None,
            None,
            config.get_n_procs(machine, cfg.WorkflowStep.GEONET),
            only_event_ids=[event_id],
            real_time=True,
            mp_sites=True,
        )

        if add_seismic_now:
            # Update with latest info
            mag, lat, lon, depth = update_eq_source_table(event_dir)
            # Send a message to slack to indicate the event is being processed
            response = send_message_to_slack(
                f"Event ID: {event_id} started processing for SeismicNow: Mag: {mag:.1f}; Depth: {depth:.1f} km; Lat: {lat:.2f}; Lon: {lon:.2f}",
            )
            message_ts = response["ts"]

        # Tectonic types
        flatfile_dir = file_structure.get_flatfile_dir(event_dir)
        eq_source_ffp = (
            flatfile_dir
            / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_GEONET
        )
        eq_tect_domain_ffp = (
            flatfile_dir
            / file_structure.PreFlatfileNames.EARTHQUAKE_SOURCE_TABLE_TECTONIC
        )
        run_nzgmdb.merge_tect_domain(
            eq_source_ffp,
            eq_tect_domain_ffp,
            config.get_n_procs(machine, cfg.WorkflowStep.TEC_DOMAIN),
        )
        # Process the records
        run_nzgmdb.process_records(
            event_dir, n_procs=config.get_n_procs(machine, cfg.WorkflowStep.PROCESS)
        )
        # Run IM_calculation
        im_dir = file_structure.get_im_dir(event_dir)
        im_dir.mkdir(parents=True, exist_ok=True)
        intensity_measures = [
            ims.IM.PGA,
            ims.IM.PGV,
            ims.IM.CAV,
            ims.IM.CAV5,
            ims.IM.AI,
            ims.IM.Ds575,
            ims.IM.Ds595,
            ims.IM.pSA,
        ]
        run_nzgmdb.run_im_calculation(
            event_dir,
            n_procs=config.get_n_procs(machine, cfg.WorkflowStep.IM),
            intensity_measures=intensity_measures,
        )
        # Merge results
        run_nzgmdb.merge_im_results(im_dir, flatfile_dir, None, None)
        # Calculate distances
        run_nzgmdb.distances.calc_distances(
            event_dir, config.get_n_procs(machine, cfg.WorkflowStep.DISTANCES)
        )
        # Merge the flatfiles
        run_nzgmdb.merge_flat_files(event_dir)

    except custom_errors.NoStationsError:
        print(f"Event {event_id} has no stations, skipping")
        # Remove the event directory
        shutil.rmtree(event_dir)
        return False

    if add_seismic_now:
        # Define the URL for the endpoint
        url = f"{SEISMIC_NOW_URL}?earthquake_id={event_id}"

        # Send a POST request to the endpoint
        response = requests.post(url)

        # Check the response status
        if response.status_code == 200:
            print("Event added successfully")
            # Get updated values
            mag, lat, lon, depth = update_eq_source_table(event_dir)
            # Add a new message to slack
            response = reply_to_message_on_slack(
                message_ts,
                f"Event ID: {event_id} added to SeismicNow (Basic Processing): Mag: {mag:.1f}; Depth: {depth:.1f} km; Lat: {lat:.2f}; Lon: {lon:.2f}",
            )
            message_ts = response["ts"]

        else:
            print(f"Failed to add event. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            # Add a new message to slack
            reply_to_message_on_slack(
                message_ts,
                f"Failed to add event {event_id} to SeismicNow, SW team investigate",
            )
            return False

    # Continue the rest of the pipeline
    run_phasenet_script_ffp = (
        Path(__file__).parent.parent / "phase_arrival/run_phasenet.py"
    )
    phase_n_procs = config.get_n_procs(machine, cfg.WorkflowStep.PHASE_TABLE)
    run_nzgmdb.make_phase_arrival_table(
        event_dir,
        flatfile_dir,
        run_phasenet_script_ffp,
        conda_sh,
        gmc_activate,
        phase_n_procs,
    )

    phase_table_path = (
        flatfile_dir / file_structure.PreFlatfileNames.PHASE_ARRIVAL_TABLE
    )
    snr_fas_output_dir = file_structure.get_snr_fas_dir(event_dir)
    run_nzgmdb.calculate_snr(
        event_dir,
        ko_matrix_path,
        phase_table_path,
        flatfile_dir,
        snr_fas_output_dir,
        config.get_n_procs(machine, cfg.WorkflowStep.SNR),
    )

    waveform_dir = file_structure.get_waveform_dir(event_dir)
    run_nzgmdb.calc_fmax(
        event_dir,
        flatfile_dir,
        waveform_dir,
        snr_fas_output_dir,
        config.get_n_procs(machine, cfg.WorkflowStep.FMAX),
    )

    run_gmc.run_gmc_processing(
        event_dir,
        gm_classifier_dir,
        ko_matrix_path,
        conda_sh,
        gmc_activate,
        gmc_predict_activate,
        config.get_n_procs(machine, cfg.WorkflowStep.GMC),
    )

    run_nzgmdb.process_records(
        event_dir, n_procs=config.get_n_procs(machine, cfg.WorkflowStep.PROCESS)
    )

    run_nzgmdb.run_im_calculation(
        event_dir,
        n_procs=config.get_n_procs(machine, cfg.WorkflowStep.IM),
        intensity_measures=intensity_measures,
    )

    run_nzgmdb.merge_im_results(im_dir, flatfile_dir, None, None)

    run_nzgmdb.merge_flat_files(event_dir)

    if add_seismic_now:
        # Get updated values
        mag, lat, lon, depth = update_eq_source_table(event_dir)
        # Reply to the slack message for final results
        reply_to_message_on_slack(
            message_ts,
            f"Event ID: {event_id} completed final processing: Mag: {mag:.1f}; Depth: {depth:.1f} km; Lat: {lat:.2f}; Lon: {lon:.2f}",
        )

    return True


@cli.from_docstring(app)
def poll_earthquake_data(
    main_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    gm_classifier_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    conda_sh: Annotated[Path, typer.Argument(exists=True, file_okay=True)],
    gmc_activate: Annotated[str, typer.Argument()],
    gmc_predict_activate: Annotated[str, typer.Argument()],
    ko_matrix_path: Annotated[
        Path | None, typer.Argument(exists=True, file_okay=False)
    ],
    add_seismic_now: Annotated[bool, typer.Option(is_flag=True)] = False,
    machine: Annotated[
        cfg.MachineName,
        typer.Option(
            case_sensitive=False,
        ),
    ] = cfg.MachineName.LOCAL,
):
    """
    Poll earthquake data, process events, and run the NZGMDB pipeline for real-time data.

    Parameters
    ----------
    main_dir : Path
        The main directory for earthquake event data.
    gm_classifier_dir : Path
        Directory for gm_classifier.
    conda_sh : Path
        Path to the conda.sh script for environment activation.
    gmc_activate : str
        Command to activate gmc environment.
    gmc_predict_activate : str
        Command to activate gmc_predict environment.
    ko_matrix_path : Path, optional
        Path to the KO matrix directory (default is None).
    add_seismic_now : bool, optional
        Whether to add the event to SeismicNow (default is False).
    machine : cfg.MachineName, optional
        The machine name to use for the number of processes (default is cfg.MachineName.LOCAL).

    Returns
    -------
    bool
        True if polling and processing were successful, False otherwise.
    """
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
                    ko_matrix_path,
                    add_seismic_now,
                    machine,
                )

                if not result:
                    # remove the event directory
                    shutil.rmtree(event_dir)
                init_start_date = end_date

        time.sleep(60)


if __name__ == "__main__":
    app()
