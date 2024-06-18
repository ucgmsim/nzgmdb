from io import StringIO
from typing import Dict, List

import requests
import numpy as np
import pandas as pd

from nzgmdb.management import config as cfg


def fetch_github_directory_contents(
    owner: str, repo: str, path: str
) -> List[Dict[str, str]]:
    """
    Fetch the contents of a GitHub directory using the GitHub API.

    Parameters:
    ----------
    owner : str
        The owner of the GitHub repository.
    repo : str
        The name of the GitHub repository.
    path : str
        The path to the directory in the repository.

    Returns:
    -------
    json_response: List[Dict[str, str]]
        The json response of the directory contents.
    """
    config = cfg.Config()
    github_api = config.get_value("github_api")
    url = f"{github_api}/{owner}/{repo}/contents/{path}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_csv_file_urls(
    contents: List[Dict[str, str]],
    owner: str,
    repo: str,
) -> List[str]:
    """
    Recursively collect CSV file URLs from the directory contents.

    Parameters:
    ----------
    contents : List[Dict[str, str]]
        The list of directory contents.
    owner : str
        The owner of the GitHub repository.
    repo : str
        The name of the GitHub repository.
    base_url : str
        The base URL for the raw GitHub content.

    Returns:
    -------
    csv_urls: List[str]
        The list of URLs for the CSV files.
    """
    config = cfg.Config()
    base_url = config.get_value("base_url")
    csv_urls = []
    for item in contents:
        if item["type"] == "file" and item["name"].endswith(".csv"):
            csv_urls.append(f"{base_url}/{owner}/{repo}/main/{item['path']}")
        elif item["type"] == "dir":
            subdir_contents = fetch_github_directory_contents(owner, repo, item["path"])
            csv_urls.extend(get_csv_file_urls(subdir_contents, owner, repo))
    return csv_urls


def download_and_read_csv(url: str) -> pd.DataFrame:
    """
    Download a CSV file from a URL and read it into a DataFrame.

    Parameters:
    ----------
    url : str
        The URL of the CSV file.

    Returns:
    -------
    pd.DataFrame
        The DataFrame containing the CSV rupture model data.
    """
    response = requests.get(url)
    response.raise_for_status()
    csv_data = StringIO(response.text)
    return pd.read_csv(csv_data)


def get_seismic_data_from_url(
    url: str,
) -> dict:
    """
    Fetch and process the seismic data from a URL.

    Parameters:
    ----------
    url : str
        The URL of the seismic data.

    Returns:
    -------
    dict
        A dictionary containing the following keys:
        'ztor' : float
            The top depth of the rupture model.
        'dbottom' : float
            The bottom depth of the rupture model.
        'strike' : float
            The strike angle of the rupture model.
        'dip' : float
            The dip angle of the rupture model.
        'rake' : float
            The rake angle of the rupture model.
        'length' : float
            The total length of the rupture model.
        'width' : float
            The average width of the rupture model.
    """
    # Get the dataframe
    df = download_and_read_csv(url)

    # Define calculated data output variables
    length = 0
    widths = []

    # Group by segment
    segments = df.groupby("SEGMENT")
    for segment, df_seg in segments:
        # Calculate the length with the first and last points along strike
        north_1, east_1 = df_seg.iloc[0]["EASTING"], df_seg.iloc[0]["NORTHING"]
        interest_depth = df_seg.iloc[0]["DEPTH"]
        north_2, east_2 = (
            df_seg[df_seg["DEPTH"] == interest_depth].iloc[-1]["EASTING"],
            df_seg[df_seg["DEPTH"] == interest_depth].iloc[-1]["NORTHING"],
        )
        length += np.linalg.norm([north_2 - north_1, east_2 - east_1]) / 1000

        # Calculate the width
        dtop = df_seg["DEPTH"].min() / 1000
        dbottom = df_seg["DEPTH"].max() / 1000
        height = dbottom - dtop
        dip = df_seg.iloc[0]["DIP"]
        width = abs(height / np.tan(np.deg2rad(dip)))
        widths.append(width)

    # Calculate the average width
    width = np.mean(widths)

    # Determine the strike, dip, and rake
    strike = df["STRIKE"].mean()
    dip = df["DIP"].mean()
    rake = df["RAKE"].mean()

    # Determine the top and bottom depths
    ztor = df["DEPTH"].min() / 1000
    dbottom = df["DEPTH"].max() / 1000

    return {
        "ztor": ztor,
        "dbottom": dbottom,
        "strike": strike,
        "dip": dip,
        "rake": rake,
        "length": length,
        "width": width,
    }


def get_rupture_models() -> dict[str, str]:
    """
    Fetch and process the rupture models from the GeoNet GitHub repository.

    Returns:
    -------
    event_urls: dict[str, str]
        The dictionary of event IDs and their corresponding CSV file URLs.
    """
    # Define the GitHub repository details
    config = cfg.Config()
    owner = config.get_value("owner")
    repo = config.get_value("repo")
    path = config.get_value("path")
    # Fetch the directory contents from GitHub
    contents = fetch_github_directory_contents(owner, repo, path)
    # Collect all CSV file URLs
    csv_urls = get_csv_file_urls(contents, owner, repo)
    # Create the dictionary of events and their urls
    event_urls = {}
    for url in csv_urls:
        event_id = url.split("/")[-1].split("_")[0]
        event_urls[event_id] = url
    return event_urls
