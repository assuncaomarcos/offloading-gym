#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for downloading and loading the WonderProxy dataset to
derive minimum and maximum latencies between a source city and
multiple servers in selected target countries.

More details on the dataset can be found
`here <https://wonderproxy.com/blog/a-day-in-the-life-of-the-internet/>`_.

Usage example:
    To download the dataset to ./data directory, and compute the latencies
    from Montreal (id 52) to all locations in Canada and the United States:

    $ python -m offloading_gym.datasets.latency --output_dir=./data --source_id 52 \
        --destination Canada --destination "United States"
"""

from os import path, makedirs

from typing import AnyStr, List
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


def valid_id(df: pd.DataFrame, location_id: int) -> bool:
    """Tests whether the provided location id is valid."""
    return df["id"].isin([location_id]).any()


def valid_country(df: pd.DataFrame, country: AnyStr) -> bool:
    """Tests whether the provided country is valid."""
    return df["country"].isin([country]).any()


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


def create_latency_files(
    datasets: Datasets, output_dir: AnyStr, source_id: int, destinations: List[AnyStr]
) -> None:
    """Creates the files with selected server information and latencies"""

    dfs = pd.read_csv(datasets.servers)
    if not valid_id(dfs, source_id):
        raise ValueError(f"Source id {source_id} is not valid")

    for destination in destinations:
        if not valid_country(dfs, destination):
            raise ValueError(f"Destination {destination} is not valid")

    dfs = dfs[dfs["country"].isin(destinations)]
    dfs.reset_index(inplace=True)
    dfs = dfs[["id", "title", "country", "latitude", "longitude"]]
    dfs.rename(columns={"id": "destination"}, inplace=True)

    dfp = pd.read_csv(datasets.pings)
    dfp = dfp[dfp["source"] == source_id]
    dfp = dfp[dfp["destination"].isin(dfs["destination"])]
    dfp = dfp.groupby("destination").agg(
        min_latency=("min", lambda x: x.quantile(0.05)),
        max_latency=("max", lambda x: x.quantile(0.95)),
    )
    dfp.reset_index(inplace=True)
    dfl = pd.merge(dfs, dfp, on="destination")
    dfl.drop("destination", axis=1, inplace=True)
    latency_file = path.join(output_dir, "latencies.csv")
    dfl.round(decimals=4).to_csv(latency_file, index=False)
    print(f"Latency information saved as {latency_file}")


@click.command()
@click.option(
    "--output_dir", help="Directory where to store the produced data", default="./data"
)
@click.option(
    "--source_id",
    type=int,
    help="The source from which latencies will be computed",
    default=SOURCE_ID,
)
@click.option(
    "--destination",
    multiple=True,
    help="Select the locations of the provided country",
    default=DESTINATION_COUNTRIES,
)
def main(**options):
    """Entry point for executing this module."""
    output_dir = options["output_dir"]
    source_id = options["source_id"]
    destinations = options["destination"]
    cache_path = path.join(output_dir, CACHE_SUBDIR)
    makedirs(cache_path, exist_ok=True)
    datasets = fetch_data(cache_path=cache_path)
    try:
        create_latency_files(datasets, output_dir, source_id, destinations)
    except ValueError as e:
        print(str(e))
