import datetime
import io
import time
import requests
from multiprocessing import Process
from pathlib import Path
from typing import List

import pandas as pd
import requests

from nzgmdb.management import config as cfg
from nzgmdb.management import file_structure, custom_errors
from nzgmdb.scripts import run_nzgmdb


def download_earthquake_data(
    start_date: datetime, end_date: datetime, mag_filter: float
) -> pd.DataFrame:
    # Calculate the current time and the time one hour ago

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


def poll_earthquake_data():
    while True:
        init_time = time.time()
        # Get the last 2 minutes worth of data and check if there are any new events
        end_date = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
        start_date = end_date - datetime.timedelta(minutes=2, days=30)
        geonet_df = download_earthquake_data(start_date, end_date, mag_filter=4.0)
        main_dir = Path("/home/joel/local/SeismicNow/event_dir")

        gm_classifier_dir = Path("/home/joel/code/gm_classifier")
        conda_sh = Path("/home/joel/anaconda3/etc/profile.d/conda.sh")
        gmc_activate = "conda activate gmc"
        gmc_predict_activate = "conda activate gmc_predict"
        n_procs = 7
        gmc_procs = 2
        ko_matrix_path = Path(
            "/home/joel/code/IM_calculation/IM_calculation/IM/KO_matrices"
        )
        checkpoint = True

        if not geonet_df.empty:
            # Get the last event_id
            only_event_ids = [str(geonet_df["publicid"].values[0])]
            # Execute the NZGMDB pipeline in the current process
            print(f"Started process for {only_event_ids[0]}")
            event_dir = main_dir / only_event_ids[0]
            event_dir.mkdir(exist_ok=True)
            try:
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
                    checkpoint=checkpoint,
                    only_event_ids=only_event_ids,
                    real_time=True,
                )
                finished_time = time.time()
                print(f"Finished and took {finished_time - init_time} seconds")
                # Get the total time taken from the earthquake happening and then the output results calculated
                eq_source_df = pd.read_csv(
                    file_structure.get_flatfile_dir(event_dir)
                    / file_structure.FlatfileNames.EARTHQUAKE_SOURCE_TABLE
                )
                eq_time = pd.to_datetime(eq_source_df["datetime"].values[0])
                print(
                    f"Total time taken: {finished_time - eq_time.timestamp()} seconds"
                )

                # Define the URL for the endpoint
                url = f"https://quakecoresoft.canterbury.ac.nz/seismicnow/api/earthquakes/add?earthquake_id={only_event_ids[0]}"

                # Send a GET request to the endpoint
                response = requests.post(url)

                # Check the response status
                if response.status_code == 200:
                    print("Event added successfully")
                else:
                    print(f"Failed to add event. Status code: {response.status_code}")
                    print(f"Response: {response.text}")
            except custom_errors.NoStations as e:
                print(e)
                print("Skip event as no stations were found")
        print("Sleeping for 1 minute")
        time.sleep(60)


if __name__ == "__main__":
    poll_earthquake_data()
