# from obspy.clients.fdsn import Client as FDSN_Client
# from obspy import UTCDateTime
# from functools import partial
# import multiprocessing
# from obspy.clients.fdsn.header import FDSNNoDataException
# from obspy.io.mseed import InternalMSEEDError, ObsPyMSEEDFilesizeTooSmallError
# import http.client
# import requests
#
#
# def do_something(station, client):
#     print(f"{station}")
#     start_time = UTCDateTime(2022, 1, 2, 1, 58, 18, 793766)
#     end_time = UTCDateTime(2022, 1, 2, 1, 59, 6, 120095)
#     # Get a waveform example
#     try:
#         st = client.get_waveforms(
#             network="NZ",
#             station=station,
#             location="*",
#             channel="HN?,BN?",
#             starttime=start_time,
#             endtime=end_time,
#         )
#     except FDSNNoDataException:
#         return None
#     except ObsPyMSEEDFilesizeTooSmallError:
#         return None
#     except (http.client.IncompleteRead, InternalMSEEDError):
#         return None
#
#     # Also send a basic request to get unique sites
#     requst_url = (
#         "https://quakecoresoft.canterbury.ac.nz/simulation_data/api/meta/unique_sites"
#     )
#     response = requests.get(requst_url)
#
#     return st, response
#
#
# n_procs = 6
# sites = [
#     "BCOF",
#     "CSBF",
#     "DCZ",
#     "DECF",
#     "GLNS",
#     "MANS",
#     "MLZ",
#     "MOSS",
#     "MSZ",
#     "MSZS",
#     "ORPS",
#     "PYZ",
#     "QTPS",
#     "RLNS",
#     "RRKS",
#     "SCTS",
#     "SECF",
#     "TAFS",
#     "WHZ",
# ]
# client_NZ = FDSN_Client("GEONET")
#
# # Repeat the sites by 100 times
# sites = sites * 1
#
# with multiprocessing.Pool(n_procs) as p:
#     results = p.map(
#         partial(
#             do_something,
#             client=client_NZ,
#         ),
#         sites,
#     )
#
# print("Completed")
import logging
import multiprocessing
import time
import sys
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import UTCDateTime
from functools import partial
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.io.mseed import InternalMSEEDError, ObsPyMSEEDFilesizeTooSmallError
import http.client
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def task_wrapper(args):
    """
    Wrapper function to execute tasks and capture any unhandled exceptions.
    """
    station, client = args
    try:
        result = do_something(station, client)
        return {"station": station, "result": result, "status": "success"}
    except FDSNNoDataException:
        return {
            "station": station,
            "result": None,
            "status": "failure",
            "error": "No data available",
        }
    except ObsPyMSEEDFilesizeTooSmallError:
        return {
            "station": station,
            "result": None,
            "status": "failure",
            "error": "MSEED file size too small",
        }
    except (http.client.IncompleteRead, InternalMSEEDError):
        return {
            "station": station,
            "result": None,
            "status": "failure",
            "error": "Incomplete read error",
        }
    except Exception as e:
        logging.exception(f"Unhandled exception in processing station {station}: {e}")
        return {
            "station": station,
            "result": None,
            "status": "failure",
            "error": str(e),
        }


def do_something(station, client):
    """
    Main task function to process each station.
    """
    # logging.info(f"Processing station: {station}")
    start_time = UTCDateTime(2022, 1, 2, 1, 58, 18, 793766)
    end_time = UTCDateTime(2022, 1, 2, 1, 59, 6, 120095)
    # Get a waveform example
    st = client.get_waveforms(
        network="NZ",
        station=station,
        location="*",
        channel="HN?,BN?",
        starttime=start_time,
        endtime=end_time,
    )
    # Send a basic request to get unique sites
    request_url = (
        "https://quakecoresoft.canterbury.ac.nz/simulation_data/api/meta/unique_sites"
    )
    response = requests.get(request_url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return {"waveform": st, "response": response.json()}


def success_callback(result):
    """
    Callback function executed upon successful completion of a task.
    """
    station = result["station"]
    # logging.info(f"Successfully completed processing for station: {station}")


def error_callback(error):
    """
    Callback function executed upon failure of a task.
    """
    logging.error(f"Error occurred: {error}")


def monitor_pool(pool):
    """
    Monitor the health of worker processes in the pool.
    """
    while True:
        alive_workers = [p.is_alive() for p in pool._pool]
        if not all(alive_workers):
            dead_workers = [i for i, alive in enumerate(alive_workers) if not alive]
            logging.error(
                f"Worker processes {dead_workers} have terminated unexpectedly."
            )
            break
        time.sleep(5)  # Check every 5 seconds


def main():
    n_procs = 20
    sites = [
        "BCOF",
        "CSBF",
        "DCZ",
        "DECF",
        "GLNS",
        "MANS",
        "MLZ",
        "MOSS",
        "MSZ",
        "MSZS",
        "ORPS",
        "PYZ",
        "QTPS",
        "RLNS",
        "RRKS",
        "SCTS",
        "SECF",
        "TAFS",
        "WHZ",
    ] * 100
    client_NZ = FDSN_Client("GEONET")

    task_args = [(site, client_NZ) for site in sites]

    with multiprocessing.Pool(processes=n_procs) as pool:
        results = []
        for args in task_args:
            res = pool.apply_async(
                task_wrapper,
                args=(args,),
                callback=success_callback,
                error_callback=error_callback,
            )
            results.append(res)

        # Periodic check for worker process health
        while not all([r.ready() for r in results]):
            alive_workers = [p.is_alive() for p in pool._pool]
            if not all(alive_workers):
                dead_workers = [i for i, alive in enumerate(alive_workers) if not alive]
                logging.error(
                    f"Worker processes {dead_workers} have terminated unexpectedly."
                )
                break
            time.sleep(5)  # Check every 5 seconds

        pool.close()
        pool.join()

        # Process results after completion
        # for res in results:
        #     result = res.get()
        #     if result["status"] == "failure":
        #         logging.error(
        #             f"Task for station {result['station']} failed with error: {result.get('error')}"
        #         )
        #     else:
        #         logging.info(
        #             f"Task for station {result['station']} completed successfully."
        #         )

    logging.info("All tasks have been processed.")


if __name__ == "__main__":
    main()
