#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List

import pandas as pd

from importlib.resources import files

from .typing import (
    CloudSite,
    Coordinate,
    Interval
)

def load_latency_info() -> List[CloudSite]:
    with files(__package__).joinpath("latencies.csv").open() as lat_file:
        df = pd.read_csv(lat_file)

    sites = []
    for _, row in df.iterrows():
        site = CloudSite(
            title=row['title'],
            country=row['country'],
            location=Coordinate(lat=row['latitude'], long=row['longitude']),
            latency=Interval(min=row['min_latency'], max=row['max_latency'])
        )
        sites.append(site)
    return sites


cloud_sites = load_latency_info()