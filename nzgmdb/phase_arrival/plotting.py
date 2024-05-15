"""
    Contains functions for plotting
    phase arrival times on waveforms
    from mseed files.
"""

import functools
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import obspy
import pandas as pd

from nzgmdb.management import file_structure


def plot_phase_arrivals_on_mseed_waveforms(
    mseed_file: Path, phase_arrival_table: Path, output_dir: Path
):
    """
    Plots waveform and phase arrival times.

    Parameters
    ----------
    mseed_file: Path
        Path to the mseed file containing waveforms.
    phase_arrival_table: Path
        Path to the phase arrival table.
    output_dir: Path
        Output directory.
    """
    phase_arrival_df = pd.read_csv(phase_arrival_table)

    # Get the event ID (evid) from the mseed file
    evid = file_structure.get_event_id_from_mseed(mseed_file)

    # Read the mseed
    mseed = obspy.read(str(mseed_file))

    # Lookup phase arrival times in the table
    phase_arrival_times = phase_arrival_df.loc[
        (phase_arrival_df["evid"] == evid)
        & (phase_arrival_df["sta"] == mseed[0].stats.station)
        & (phase_arrival_df["chan"] == mseed[0].stats.channel[:2])
        & (phase_arrival_df["loc"] == mseed[0].stats.location)
    ]["datetime"]

    # Creating a list of phase arrival times,
    # formatted for plotting with matplotlib
    arrival_times_as_list = [
        obspy.UTCDateTime(arrival_time).matplotlib_date
        for arrival_time in phase_arrival_times.values
    ]

    if len(arrival_times_as_list) > 0:

        # Generate most of the plot with Obspy.
        # Using handle=True returns the plot
        fig = mseed.plot(handle=True)

        # Add a title above the top subplot
        fig.axes[0].set_title(mseed_file.stem)

        # Plot the arrival phases on all subplots
        # using arbitrary large ymin and ymax to
        # span the expected vertical range
        for ax in fig.axes:
            ax.vlines(arrival_times_as_list, ymin=-1e6, ymax=1e6, linestyle="--")
        fig.savefig(output_dir / f"{mseed_file.stem}.png")
        plt.close()


def batch_plot_phase_arrivals(
    main_dir: Path, phase_arrival_table: Path, output_dir: Path, n_procs: int
):
    """
    Plots waveform and phase arrival times
    for a batch of mseed files.

    Parameters
    ----------
    n_procs
    main_dir : Path
        The main directory of the NZGMDB results (highest level directory)
        (glob is used to find all mseed files recursively).
    phase_arrival_table: Path
        Path to the phase arrival table.
    output_dir: Path
        Output directory.
    """

    # Find all mseed files recursively
    mseed_files = list(main_dir.glob("**/*.mseed"))
    # Initialize a multiprocessing Pool
    with multiprocessing.Pool(processes=n_procs) as pool:
        # Map the plotting function to the file chunks
        pool.map(
            functools.partial(
                plot_phase_arrivals_on_mseed_waveforms,
                phase_arrival_table=phase_arrival_table,
                output_dir=output_dir,
            ),
            mseed_files,
        )
