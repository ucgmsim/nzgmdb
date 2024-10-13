import datetime
import time
from pathlib import Path
from typing import List
import requests
import pandas as pd
import io
from multiprocessing import Process
from nzgmdb.management import config as cfg


import datetime
import requests
import pandas as pd
import io
from nzgmdb.management import config as cfg
from nzgmdb.scripts import run_nzgmdb


def download_earthquake_data_last_hour(
    start_date: datetime, end_date: datetime
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

    return df


def poll_earthquake_data():
    while True:
        init_time = time.time()
        # Get the last 2 minutes worth of data and check if there are any new events
        end_date = datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
        start_date = end_date - datetime.timedelta(minutes=2)
        geonet_df = download_earthquake_data_last_hour(start_date, end_date)
        main_dir = Path("/home/joel/local/gmdb/real_time/testing")

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

            # Now we need to execute the full NZGMDB pipeline as a new process
            # process = Process(
            #     target=run_nzgmdb.run_full_nzgmdb,
            #     args=(main_dir, start_date, end_date, gm_classifier_dir, conda_sh, gmc_activate, gmc_predict_activate,
            #           gmc_procs, n_procs),
            #     kwargs={'ko_matrix_path': ko_matrix_path, 'checkpoint': checkpoint, 'only_event_ids': only_event_ids}
            # )
            # process.start()
            # Execute the NZGMDB pipeline in the current process
            print(f"Started process for {only_event_ids[0]}")
            event_dir = main_dir / only_event_ids[0]
            event_dir.mkdir(exist_ok=True)
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
            print(f"Finished and took {time.time() - init_time} seconds")
        # print("Sleeping for 1 minute")
        time.sleep(60)


if __name__ == "__main__":
    poll_earthquake_data()
