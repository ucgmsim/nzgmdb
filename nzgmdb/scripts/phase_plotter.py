"""
    Contains function scripts to plot
    phase arrival times on mseed waveforms
"""

from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.phase_arrival.phase_arrival_plotting_functions import (
    plot_phase_arrivals_on_mseed_waveforms,
    batch_plot_phase_arrivals,
)

app = typer.Typer()


@app.command(help="Plots waveform and phase arrival times.")
def make_plot(
    mseed_file: Annotated[
        Path,
        typer.Argument(
            help="Path to the mseed file containing waveforms.",
            exists=True,
        ),
    ],
    phase_arrival_table: Annotated[
        Path,
        typer.Argument(
            help="Path to the phase arrival table.",
            exists=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the plot in.",
            exists=True,
        ),
    ],
):
    plot_phase_arrivals_on_mseed_waveforms(mseed_file, phase_arrival_table, output_dir)


@app.command(help="Plots waveform and phase arrival times for a batch of mseed files.")
def batch_plot(
    main_dir: Annotated[
        Path,
        typer.Argument(
            help="The main directory of the NZGMDB results (highest level directory) "
            "(glob is used to find all mseed files recursively).",
            exists=True,
        ),
    ],
    phase_arrival_table: Annotated[
        Path,
        typer.Argument(
            help="The phase arrival table to load.",
            exists=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the plot in.",
            exists=True,
        ),
    ],
    n_procs: Annotated[int, typer.Option(help="Number of processes to use")] = 1,
):
    batch_plot_phase_arrivals(main_dir, phase_arrival_table, output_dir, n_procs)


if __name__ == "__main__":
    app()
