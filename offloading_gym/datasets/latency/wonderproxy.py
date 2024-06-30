#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for downloading and loading the WonderProxy dataset to
derive minimum and maximum latencies between a source city and
multiple servers in selected target countries.

Usage:
    To download the dataset and generate the latency files in the ./data directory:
    $ python -m offloading_gym.datasets.latency --output_dir=./data
"""

from os import path, makedirs

from typing import AnyStr
from collections import namedtuple

import click
import pandas as pd

from offloading_gym.datasets.util import download_file, decompress_gz

CACHE_SUBDIR = "downloaded"
PINGS_URL = (
    "https://wp-public.s3.amazonaws.com/pings/pings-2020-07-19-2020-07-20.csv.gz"
)
SERVERS_URL = "https://wp-public.s3.amazonaws.com/pings/servers-2020-07-19.csv"
DESTINATION_COUNTRIES = ["United States", "Canada"]
SOURCE_ID = 52  # The id for Montreal

Datasets = namedtuple("Datasets", ["servers", "pings"])


def fetch_data(cache_path: AnyStr) -> Datasets:
    """Download the WonderProxy dataset and decompress any gz file."""

    def download_if_not_exists(fetch_url) -> AnyStr:
        save_file = path.join(cache_path, path.basename(fetch_url))
        _, ext = path.splitext(save_file)

        if not path.exists(save_file):
            download_file(fetch_url, save_file)

        if ext.lower() == ".gz":
            decompress_gz(save_file, cache_path)
            save_file, _ = path.splitext(save_file)

        print(f"File downloaded and saved as {save_file}")
        return save_file

    return Datasets(
        servers=download_if_not_exists(SERVERS_URL),
        pings=download_if_not_exists(PINGS_URL),
    )


def create_latency_files(datasets: Datasets, output_dir: AnyStr):
    """Creates the files with selected server information and latencies"""

    dfs = pd.read_csv(datasets.servers)
    dfs = dfs[dfs["country"].isin(DESTINATION_COUNTRIES)]
    dfs.reset_index(inplace=True)
    dfs = dfs[["id", "title", "country", "latitude", "longitude"]]

    dfp = pd.read_csv(datasets.pings)
    dfp = dfp[dfp["source"] == SOURCE_ID]
    dfp = dfp[dfp["destination"].isin(dfs["id"])]
    dfp = dfp.groupby("destination").agg(
        min_latency=("min", lambda x: x.quantile(0.05)),
        max_latency=("max", lambda x: x.quantile(0.95)),
    )
    dfp.reset_index(inplace=True)
    dfs.to_csv(path.join(output_dir, "locations.csv"), index=False)
    dfp.round(decimals=4).to_csv(path.join(output_dir, "latencies.csv"), index=False)


@click.command()
@click.option(
    "--output_dir", help="Directory where to store the produced data", default="./data"
)
def main(**options):
    """Entry point for executing this module."""
    output_dir = options["output_dir"]
    cache_path = path.join(output_dir, CACHE_SUBDIR)
    makedirs(cache_path, exist_ok=True)
    datasets = fetch_data(cache_path=cache_path)
    create_latency_files(datasets, output_dir)
