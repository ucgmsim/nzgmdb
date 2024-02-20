"""
    Creates the phase arrival table from a directory of mseed files
"""
import pandas as pd
from obspy import read
from picker import p_phase_picker
import multiprocessing
import numpy as np
from pathlib import Path
from datetime import timedelta


def get_p_wave(data: np.ndarray, dt: int):
    """
    Get the P wave arrival time from the data

    Parameters
    ----------
    data : np.ndarray
        The waveform data from a single component
    dt : int
        The sample rate of the data
    """
    wftype = "SM"  # Input wftype is strong motion
    try:
        loc = p_phase_picker(data, dt, wftype)
    except Exception as e:
        print(e)
        loc = -1
    return loc


def process_mseed(mseed_file_chunk: list):
    """
    Process a chunk of mseed files and return the phase arrival data

    Parameters
    ----------
    mseed_file_chunk : list
        A list of mseed files to process
    """
    data_list = []
    for file in mseed_file_chunk:
        # Read the mseed file
        mseed_data = read(str(file))

        try:
            p_comp_000 = get_p_wave(mseed_data[0].data, mseed_data[0].stats.delta)
        except:
            p_comp_000 = -1
        try:
            p_comp_090 = get_p_wave(mseed_data[1].data, mseed_data[1].stats.delta)
        except:
            p_comp_090 = -1
        try:
            p_comp_ver = get_p_wave(mseed_data[2].data, mseed_data[2].stats.delta)
        except:
            p_comp_ver = -1

        if p_comp_ver > 1:
            # Vertical is greater than 1 use this for the record
            p_wave_loc = p_comp_ver
        else:
            # Find the max between all three components and take away 0.5
            p_wave_loc = max(p_comp_000, p_comp_090, p_comp_ver) - 0.5

        if p_wave_loc < 0:
            continue
        else:
            arid = None
            stats = mseed_data[0].stats

            # Get the datetime
            datetime = stats.starttime + timedelta(seconds=p_wave_loc)

            # Get the net
            net = stats.network

            # Get the sta
            sta = stats.station

            # Get the loc
            loc = stats.location

            # Get the chan
            chan = stats.channel

            # Get the phase
            phase = "P"

            # Get the t_res
            t_res = None

            # Get the evid
            xml_dir = file.parent.parent.parent
            # Find the name of the xml file
            xml_file = list(xml_dir.glob("*.xml"))[0]
            # Get the evid from the name without the .xml
            evid = xml_file.stem

            new_data = {
                "arid": arid,
                "datetime": datetime,
                "net": net,
                "sta": sta,
                "loc": loc,
                "chan": chan,
                "phase": phase,
                "t_res": t_res,
                "evid": evid,
            },
            data_list.append(new_data)
    return data_list


def generate_phase_arrival_table(data_dir: Path, output_dir: Path, n_procs: int):
    """
    Generate the phase arrival table from a directory of mseed files
    and save it to a csv file in the output directory

    Parameters
    ----------
    data_dir : Path
        The top directory containing the mseed files
        (glob is used to find all mseed files recursively)
    output_dir : Path
        The directory to save the phase arrival table
    n_procs : int
        The number of processes to use
    """

    # Find all mseed files recursively
    mseed_files = list(data_dir.glob("**/*.mseed"))

    # Split the mseed files into chunks based on the number of processes
    file_chunks = [
        mseed_files[i::n_procs]
        for i in range(n_procs)
    ]

    # Initialize a multiprocessing Pool
    with multiprocessing.Pool(processes=n_procs) as pool:
        # Map the reading function to the file chunks
        mseed_data_list = pool.map(process_mseed, file_chunks)

    # Create the dataframe
    df = pd.DataFrame([tup[0] for data_list in mseed_data_list for tup in data_list])

    # Save the dataframe
    df.to_csv(output_dir / "phase_arrival_table.csv", index=False)


generate_phase_arrival_table(Path("path/to/mseed/files"), Path("path/to/output/dir"), 1)

