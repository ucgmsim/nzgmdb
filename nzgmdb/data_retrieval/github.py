"""
This module contains functions for fetching data from GitHub repositories.
"""

import requests

from nzgmdb.management import config as cfg


def fetch_github_directory_contents(
    owner: str, repo: str, path: str
) -> list[dict[str, str]]:
    """
    Fetch the contents of a GitHub directory using the GitHub API.

    Parameters
    ----------
    owner : str
        The owner of the GitHub repository.
    repo : str
        The name of the GitHub repository.
    path : str
        The path to the directory in the repository.

    Returns
    -------
    list[dict[str, str]]
        The JSON response containing the directory contents.
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

    Parameters
    ----------
    contents : list[dict[str, str]]
        The list of directory contents.
    owner : str
        The owner of the GitHub repository.
    repo : str
        The name of the GitHub repository.

    Returns
    -------
    list[str]
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
