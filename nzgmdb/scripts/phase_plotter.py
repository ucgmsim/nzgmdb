"""
    Contains function scripts to plot
    phase arrival times on mseed waveforms
"""

from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.phase_arrival import plotting

app = typer.Typer()


@app.command(help="Plots waveform and phase arrival times.")
def make_plot(
    mseed_file: Annotated[
        Path,
        typer.Argument(
            help="Path to the mseed file containing waveforms.",
            exists=True,
            file_okay=True,
        ),
    ],
    phase_arrival_table: Annotated[
        Path,
        typer.Argument(
            help="Path to the phase arrival table.",
            exists=True,
            file_okay=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the plot in.",
            exists=True,
            file_okay=False,
        ),
    ],
):  # noqa: D103
    plotting.plot_phase_arrivals_on_mseed_waveforms(
        mseed_file, phase_arrival_table, output_dir
    )


@app.command(help="Plots waveform and phase arrival times for a batch of mseed files.")
def batch_plot(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (highest level directory) "
            "(glob is used to find all mseed files recursively).",
            exists=True,
            file_okay=False,
        ),
    ],
    phase_arrival_table: Annotated[
        Path,
        typer.Argument(
            help="The phase arrival table to load.",
            exists=True,
            file_okay=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the plot in.",
            file_okay=False,
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
):  # noqa: D103
    plotting.batch_plot_phase_arrivals(
        main_dir, phase_arrival_table, output_dir, n_procs
    )


@app.command(
    help="Plots a histogram of the differences in phase arrival times from picker and Geonet"
)
def plot_hist(
    phase_arrival_table: Annotated[
        Path,
        typer.Argument(
            help="Path to the phase arrival table that includes columns datetime_picker and datetime_geonet.",
            exists=True,
            file_okay=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the plot in.",
            exists=True,
            file_okay=False,
        ),
    ],
    num_bins: Annotated[int, typer.Option(help="Number of bins in the histogram")] = 50,
    dpi: Annotated[int, typer.Option(help="dpi of saved plot")] = 500,
):  # noqa: D103
    plotting.plot_time_diffs_hist(phase_arrival_table, output_dir, num_bins, dpi)


if __name__ == "__main__":
    app()
