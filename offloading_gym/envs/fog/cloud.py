#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module also loads the information on the network latency from Montreal to
all the other Wonderproxy servers in Canada and the USA to be used for modeling
the latency of IoT devices to cloud servers.

More details on the Wonderproxy dataset can be found
`here <https://wonderproxy.com/blog/a-day-in-the-life-of-the-internet/>`_.
"""

from offloading_gym.simulation.fog.typing import CloudSite, Interval

from importlib.resources import files
from typing import List

import pandas as pd

__all__ = ["cloud_sites"]


def _load_latency_info() -> List[CloudSite]:
    with files(__package__).joinpath("latencies.csv").open() as lat_file:
        df = pd.read_csv(lat_file)

    sites = []
    for _, row in df.iterrows():
        site = CloudSite(
            lat=row["latitude"],
            long=row["longitude"],
            title=row["title"],
            country=row["country"],
            latency=Interval(min=row["min_latency"], max=row["max_latency"]),
        )
        sites.append(site)
    return sites


_server_info = _load_latency_info()


def cloud_sites() -> List[CloudSite]:
    """Returns the latency information as a list of `CloudSite` objects."""
    return _server_info
