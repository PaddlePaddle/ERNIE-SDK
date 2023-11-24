from __future__ import annotations

import requests


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith("http://") or path.startswith("https://")


def url_file_exists(url: str, headers: dict) -> bool:
    """check whether the url file exists

        refer to: https://stackoverflow.com/questions/2486145/python-check-if-url-to-jpg-exists

    Args:
        url (str): the url of target file
        headers (dict): the header of http request

    Returns:
        bool: whether the url file exists
    """
    if not is_url(url):
        return False

    result = requests.head(url, headers=headers)
    return result.status_code == requests.codes.ok
