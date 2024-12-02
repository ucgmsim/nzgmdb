"""
Contains functions for plotting
phase arrival times on waveforms
from mseed files.
"""

import functools
import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    if "datetime" in phase_arrival_df.keys():
        # Picker P-phase and GeoNet S-phase (if present)
        plot_subdir = output_dir / "picker_p_and_geonet_s"

        phase_arrival_times = phase_arrival_df.loc[
            (phase_arrival_df["evid"] == evid)
            & (phase_arrival_df["sta"] == mseed[0].stats.station)
            & (phase_arrival_df["chan"] == mseed[0].stats.channel[:2])
            & (phase_arrival_df["loc"] == mseed[0].stats.location)
        ]["datetime"]

        # "r" color will be ignored if there is only one phase arrival time
        vline_colors = ["b", "r"]

    else:
        # All phase picks from Picker and GeoNet
        plot_subdir = output_dir / "all_phase_from_picker_and_geonet"

        picker_phase_arrival_times = phase_arrival_df.loc[
            (phase_arrival_df["evid"] == evid)
            & (phase_arrival_df["sta"] == mseed[0].stats.station)
            & (phase_arrival_df["chan"] == mseed[0].stats.channel[:2])
            & (phase_arrival_df["loc"] == mseed[0].stats.location)
        ]["datetime_picker"]

        geonet_phase_arrival_times = phase_arrival_df.loc[
            (phase_arrival_df["evid"] == evid)
            & (phase_arrival_df["sta"] == mseed[0].stats.station)
            & (phase_arrival_df["chan"] == mseed[0].stats.channel[:2])
            & (phase_arrival_df["loc"] == mseed[0].stats.location)
        ]["datetime_geonet"]

        phase_arrival_times = pd.concat(
            [picker_phase_arrival_times, geonet_phase_arrival_times]
        )

        # plot picker as blue and geonet as red
        vline_colors = ["b"] * len(picker_phase_arrival_times)
        vline_colors.extend(["r"] * len(geonet_phase_arrival_times))

    arrival_times_as_list = []
    for arrival_time in phase_arrival_times:
        if isinstance(arrival_time, str):
            arrival_times_as_list.append(
                obspy.UTCDateTime(arrival_time).matplotlib_date
            )
        else:
            arrival_times_as_list.append(np.nan)

    if arrival_times_as_list:
        # Create the output subdirectory if it doesn't already exist
        plot_subdir.mkdir(parents=True, exist_ok=True)

        # Generate most of the plot with Obspy.
        # Using handle=True returns the plot
        fig = mseed.plot(handle=True)

        # Add a title above the top subplot
        fig.axes[0].set_title(
            f"{mseed_file.stem}, Picker is blue, GeoNet (if present) is red"
        )

        # Plot the arrival phases on all subplots
        for ax in fig.axes:
            ax.vlines(
                arrival_times_as_list,
                ymin=ax.get_ylim()[0],
                ymax=ax.get_ylim()[1],
                linestyle="--",
                colors=vline_colors,
            )

        fig.savefig(plot_subdir / f"{mseed_file.stem}.png")
        plt.close()


def batch_plot_phase_arrivals(
    main_dir: Path, phase_arrival_table: Path, output_dir: Path, n_procs: int
):
    """
    Plots waveform and phase arrival times
    for a batch of mseed files.

    Parameters
    ----------
    main_dir : Path
        The main directory of the NZGMDB results (highest level directory)
        (glob is used to find all mseed files recursively).
    phase_arrival_table: Path
        Path to the phase arrival table.
    output_dir: Path
        Output directory.
    n_procs : int
        The number of processes to use
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


def plot_time_diffs_hist(
    phase_arrival_table: Path, output_dir: Path, num_bins: int = 50, dpi: int = 500
):
    """

    Plots a histogram of the differences in phase arrival times from our picker
    code and Geonet.

    Parameters
    ----------
    phase_arrival_table: Path
        Path to the phase arrival table that includes columns
        datetime_picker and datetime_geonet.
    output_dir: Path
         Output directory.
    num_bins: int, optional
        Number of bins to use in the histogram
    """

    phase_arrival_df = pd.read_csv(phase_arrival_table)

    time_diffs = phase_arrival_df["picker_time_minus_geonet_time_secs"]

    plt.hist(
        time_diffs[~np.isnan(time_diffs)],
        bins=num_bins,
        alpha=0.75,
        color="blue",
        edgecolor="black",
    )

    counts, bin_edges = np.histogram(time_diffs[~np.isnan(time_diffs)], bins=num_bins)
    total_points = counts.sum()

    plt.xlabel("Picker time - Geonet time (seconds)")
    plt.ylabel("count")
    print(f"Total number of points used in the histogram: {total_points}")
    plt.title(f"{total_points} arrival times from both Picker and GeoNet")

    plt.savefig(output_dir / "histogram.png", dpi=dpi)
    plt.close()
