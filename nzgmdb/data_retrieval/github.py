from io import StringIO

import pandas as pd
import requests

from nzgmdb.management import config as cfg


def fetch_github_directory_contents(
    owner: str, repo: str, path: str
) -> list[dict[str, str]]:
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
    json_response: list[dict[str, str]]
        The json response of the directory contents.
    """
    config = cfg.Config()
    github_api = config.get_value("github_api")
    url = f"{github_api}/{owner}/{repo}/contents/{path}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_csv_file_urls(
    contents: list[dict[str, str]],
    owner: str,
    repo: str,
) -> list[str]:
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
    csv_urls: list[str]
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
        The DataFrame containing the CSV data.
    """
    response = requests.get(url)
    response.raise_for_status()
    csv_data = StringIO(response.text)
    return pd.read_csv(csv_data)
