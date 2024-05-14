import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import obspy
import pandas as pd

from nzgmdb.management import file_structure


def plot_phase_arrivals_on_mseed_waveforms(
    mseed_file: Path, phase_arrival_table: Path, output_dir: Path
):
    phase_arrival_df = pd.read_csv(phase_arrival_table)

    # Get the event ID (evid) from the mseed file
    evid = file_structure.get_event_id_from_mseed(mseed_file)

    # Read the mseed
    mseed = obspy.read(str(mseed_file))

    # Get the phase arrival times
    phase_arrival_times = phase_arrival_df.loc[
        (phase_arrival_df["evid"] == evid)
        & (phase_arrival_df["sta"] == mseed[0].stats.station)
        & (phase_arrival_df["chan"] == mseed[0].stats.channel[:2])
        & (phase_arrival_df["loc"] == mseed[0].stats.location)
    ]["datetime"]

    arrival_times_as_list = []
    for arrival_time in phase_arrival_times.values:
        arrival_times_as_list.append(obspy.UTCDateTime(arrival_time).matplotlib_date)

    if len(arrival_times_as_list) > 0:

        # Use Obspy to generate most of the plot. Using handle=True returns the
        # Python plot handle enabling further modification
        fig = mseed.plot(handle=True)

        # Add a title above the top subplot
        fig.axes[0].set_title(mseed_file.stem)

        # Plot the arrival phases on all subplots
        for ax in fig.axes:
            ax.vlines(arrival_times_as_list, ymin=-1e6, ymax=1e6, linestyle="--")
        fig.savefig(output_dir / f"{mseed_file.stem}.png")
        plt.close()


#############################################################################

mseed_file_chunk: list

def batch_plot_phase_arrivals(
    main_dir: Path, phase_arrival_table: Path, output_dir: Path, n_procs: int
):

    # Find all mseed files recursively
    mseed_files = list(main_dir.glob("**/*.mseed"))

    # Split the mseed files into chunks based on the number of processes
    file_chunks = [mseed_files[i::n_procs] for i in range(n_procs)]

    # Initialize a multiprocessing Pool
    with multiprocessing.Pool(processes=n_procs) as pool:
        # Map the reading function to the file chunks
        mseed_data_list = pool.map(process_mseed, file_chunks)
