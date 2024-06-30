#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions common to modules in this package.
"""

import zipfile
from os import path
from typing import List, AnyStr
import requests
import gzip
import shutil

TIMEOUT = 2.0


def download_file(url, save_path):
    """
    Downloads a file and saves it on disk
    Args:
        url: the URL to download the file from
        save_path: the path to save the file

    Returns: None
    Raises: RequestException if any error downloading the file occurs
    """
    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            file.write(response.content)
    except requests.exceptions.RequestException as exception:
        raise exception


def zip_files(zip_filename: AnyStr, to_zip: List[AnyStr]) -> None:
    """
    Zips a set of files.

    Args:
        zip_filename: the name of the zip file
        to_zip: the list of files to include into the zip file

    Returns:
        None
    """
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        for file in to_zip:
            zipf.write(file, path.basename(file))


def decompress_gz(gz_file_path: AnyStr, output_dir: AnyStr) -> List[AnyStr]:
    """
    Uncompress a gzipped file.

    Args:
        gz_file_path: the path to the gzipped file
        output_dir: the directory to decompress the gzipped file

    Returns:
        None
    """
    basename, _ = path.splitext(path.basename(gz_file_path))
    output_path = path.join(output_dir, basename)

    with gzip.open(gz_file_path, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
