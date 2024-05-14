from pathlib import Path
from typing import Annotated

import typer

from nzgmdb.phase_arrival.plot_phase_arrival import (
    plot_phase_arrivals_on_mseed_waveforms,
)

app = typer.Typer()


@app.command(help="help text here")
def make_plot(
    mseed_file: Annotated[
        Path,
        typer.Argument(
            help="The mseed file to load",
            exists=True,
        ),
    ],
    phase_arrival_table: Annotated[
        Path,
        typer.Argument(
            help="The phase arrival table to load",
            exists=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="The directory to save the plot",
        ),
    ],
):
    plot_phase_arrivals_on_mseed_waveforms(mseed_file, phase_arrival_table, output_dir)


@app.command(help="help text here")
def call_make_plot():
    pass


if __name__ == "__main__":
    app()
