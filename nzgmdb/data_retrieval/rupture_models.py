"""
This module contains functions for fetching and processing rupture models from the GeoNet GitHub repository.
"""

from typing import TypedDict

import numpy as np
import pandas as pd

from nzgmdb.data_retrieval import github
from nzgmdb.management import config as cfg
from qcore import geo


class RuptureModel(TypedDict):
    """
    A dictionary representing a rupture model.
    """

    ztor: float
    """The top-depth of the rupture model."""
    dbottom: float
    """The bottom-depth of the rupture model."""
    strike: float
    """The strike angle of the rupture model."""
    dip: float
    """The dip angle of the rupture model."""
    rake: float
    """The dip angle of the rupture model."""
    length: float
    """The total length of the rupture model."""
    width: float
    """The average width of the rupture model."""


def get_seismic_data_from_url(url: str) -> dict:
    """
    Fetch and process the seismic data from a URL.

    Parameters
    ----------
    url : str
        The URL from which the seismic data is retrieved.

    Returns
    -------
    RuptureModel
        The rupture model read from `url`.
    """
    # Get the dataframe
    df = pd.read_csv(url)

    # Define calculated data output variables
    length = 0
    widths = []

    # Group by segment
    segments = df.groupby("SEGMENT")
    for segment, df_seg in segments:
        # Calculate the length with the first and last points along strike
        try:
            north_1, east_1 = df_seg.iloc[0]["NORTHING"], df_seg.iloc[0]["EASTING"]
            interest_depth = df_seg.iloc[0]["DEPTH"]
            north_2, east_2 = (
                df_seg[df_seg["DEPTH"] == interest_depth].iloc[-1]["NORTHING"],
                df_seg[df_seg["DEPTH"] == interest_depth].iloc[-1]["EASTING"],
            )
            length += np.linalg.norm([north_2 - north_1, east_2 - east_1]) / 1000
        except KeyError:
            # This means Northing was not found in the dataframe and so need to compute the distance based on lat lon
            lat1, lon1 = df_seg.iloc[0]["lat"], df_seg.iloc[0]["lon"]
            lat2, lon2 = df_seg.iloc[-1]["lat"], df_seg.iloc[-1]["lon"]

            length += geo.ll_dist(lat1, lon1, lat2, lon2)

        # Calculate the width
        dtop = df_seg["DEPTH"].min() / 1000
        dbottom = df_seg["DEPTH"].max() / 1000
        height = dbottom - dtop
        dip = df_seg.iloc[0]["DIP"]
        width = abs(height / np.tan(np.deg2rad(dip)))
        widths.append(width)

    # Calculate the average width
    width = float(np.mean(widths))

    # Determine the strike, dip, and rake
    strike = df["STRIKE"].mean()
    dip = df["DIP"].mean()
    rake = df["RAKE"].mean()

    # Determine the top and bottom depths
    ztor = df["DEPTH"].min() / 1000
    dbottom = df["DEPTH"].max() / 1000

    # Create an instance of RuptureModel
    rupture_model: RuptureModel = {
        "ztor": ztor,
        "dbottom": dbottom,
        "strike": strike,
        "dip": dip,
        "rake": rake,
        "length": length,
        "width": width,
    }

    return rupture_model


def get_rupture_models() -> dict[str, str]:
    """
    Fetch and process the rupture models from the GeoNet GitHub repository.

    Returns
    -------
    dict[str, str]
        A dictionary where the keys are event IDs and the values are the corresponding CSV file URLs.
    """
    # Define the GitHub repository details
    config = cfg.Config()
    owner = config.get_value("owner")
    repo = config.get_value("repo")
    path = config.get_value("path")
    # Fetch the directory contents from GitHub
    contents = github.fetch_github_directory_contents(owner, repo, path)
    # Collect all CSV file URLs
    csv_urls = github.get_csv_file_urls(contents, owner, repo)
    # Create the dictionary of events and their urls
    event_urls = {}
    for url in csv_urls:
        event_id = url.split("/")[-1].split("_")[0]
        event_urls[event_id] = url
    return event_urls
