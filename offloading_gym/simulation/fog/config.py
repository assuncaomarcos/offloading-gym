#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module containing the classes required for building
the configuration of a fog environment.

This module also loads the information on the network latency from Montreal to
all the other Wonderproxy servers in Canada and the USA to be used for modeling
the latency of IoT devices to cloud servers.

More details on the Wonderproxy dataset can be found
`here <https://wonderproxy.com/blog/a-day-in-the-life-of-the-internet/>`_.
"""

from __future__ import annotations

from typing import List, NamedTuple, Union, AnyStr, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

from importlib.resources import files

import pandas as pd

BYTES_IN_MB = 2 ** 20


@dataclass(frozen=True)
class Coordinate:
    """Represents a geographical coordinate."""

    lat: float
    long: float


@dataclass(frozen=True)
class RectGeographicalArea:
    """
    Set of geographical coordinates that form a rectangular area
    on which edge resources are deployed.
    """

    northeast: Coordinate
    northwest: Coordinate
    southeast: Coordinate
    southwest: Coordinate

    def __post_init__(self):
        if not self._is_rectangle():
            raise ValueError("Set of coordinates is not a rectangular area.")

    def _is_rectangle(self):
        # Check to see if the coordinates form a rectangle
        return (
            self.northeast.lat == self.northwest.lat
            and self.southeast.lat == self.southwest.lat
            and self.northeast.long == self.southeast.long
            and self.northwest.long == self.southwest.long
        )


class ResourceType(Enum):
    IOT = "iot"
    EDGE = "edge"
    CLOUD = "cloud"

    # Python < 3.11 does not have StrEnum
    def __str__(self):
        return self.value


class ResourceConfig(NamedTuple):
    """Data type for computing resource configuration."""

    cpu_cores: List[int]
    cpu_core_speed: List[float]
    memory: List[float]


class Interval(NamedTuple):
    """Data type for a value interval."""

    min: float
    max: float


class NetworkConfig(NamedTuple):
    """Data type for computing network configuration."""

    bandwidth: Interval


GeographicalArea = Union[RectGeographicalArea, List[Coordinate]]


@dataclass(frozen=True)
class CloudSite(Coordinate):
    title: AnyStr
    country: AnyStr
    latency: Interval


@dataclass(frozen=True)
class ResourceGroupConfig:
    """
    Data type for the configuration of a group of
    resources (e.g., IoT devices, edge servers, cloud servers)
    """

    num_resources: int
    resource_config: ResourceConfig
    network_config: Union[NetworkConfig, None]
    deployment_area: Union[GeographicalArea, List[CloudSite]]


@dataclass(frozen=True)
class ComputingConfig:
    """Data type for the configuration data of a fog environment."""

    iot: ResourceGroupConfig
    edge: ResourceGroupConfig
    cloud: ResourceGroupConfig

    def num_compute_resources(self) -> int:
        """
        Returns the number of compute resources (IoT devices, edge servers,
        and cloud servers) to be created according as per this configuration
        object.

        Returns:
            The number of compute resources.
        """
        return self.iot.num_resources + self.edge.num_resources + self.cloud.num_resources

    def max_attribute_value(self, attribute: str) -> int:
        """Returns the maximum value of an attribute a resource in the environment has."""
        max_value = 0
        for group in [self.iot, self.edge, self.cloud]:
            attribute_list = getattr(group.resource_config, attribute, None)
            if attribute_list:
                max_value = max(max_value, *attribute_list)
        return max_value

    def max_number_cores(self) -> int:
        """Returns the maximum number of cores a resource in the environment has."""
        return self.max_attribute_value('cpu_cores')

    def max_core_speed(self) -> int:
        """Returns the maximum speed a compute core in the environment has."""
        return self.max_attribute_value('cpu_core_speed')

    def max_memory(self) -> int:
        """Returns the maximum memory size a compute resource in the environment has."""
        return self.max_attribute_value('memory')


@dataclass(frozen=True)
class WorkloadConfig:
    num_tasks: List[int]
    min_computing: int
    max_computing: int
    min_memory: int
    max_memory: int
    min_datasize: int
    max_datasize: int
    density_values: List[float]
    regularity_values: List[float]
    fat_values: List[float]
    ccr_values: List[float]
    jump_values: List[int]

    def as_dict(self) -> Dict[AnyStr, Any]:
        """Returns the workload configuration as a dict"""
        return asdict(self)


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


def server_info() -> List[CloudSite]:
    """Returns the latency information as a list of `CloudSite` objects."""
    return _server_info


# The following GPS coordinates roughly encompass the entire
# city of Montreal, forming a rough rectangular boundary around it.
MONTREAL_AREA = RectGeographicalArea(
    northeast=Coordinate(lat=45.7057, long=-73.4746),
    northwest=Coordinate(lat=45.7057, long=-73.9434),
    southeast=Coordinate(lat=45.3831, long=-73.4746),
    southwest=Coordinate(lat=45.3831, long=-73.9434),
)

DEFAULT_COMP_CONFIG = ComputingConfig(
    iot=ResourceGroupConfig(
        num_resources=1,
        resource_config=ResourceConfig(
            cpu_cores=[1], cpu_core_speed=[1.0], memory=[1.0]
        ),
        network_config=None,
        deployment_area=MONTREAL_AREA,
    ),
    edge=ResourceGroupConfig(
        num_resources=36,
        resource_config=ResourceConfig(
            cpu_cores=[4], cpu_core_speed=[1.5, 1.8, 2.0], memory=[1.0, 2.0, 4.0]
        ),
        network_config=NetworkConfig(bandwidth=Interval(min=10, max=12)),
        deployment_area=MONTREAL_AREA,
    ),
    cloud=ResourceGroupConfig(
        num_resources=20,
        resource_config=ResourceConfig(
            cpu_cores=[8], cpu_core_speed=[2.0, 2.6, 3.0], memory=[16.0, 24.0, 32.0]
        ),
        network_config=NetworkConfig(bandwidth=Interval(min=4, max=8)),
        deployment_area=server_info(),
    ),
)

DEFAULT_WORKLOAD_CONFIG = WorkloadConfig(
    num_tasks=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    min_computing=(10 ** 7),
    max_computing=(3 * 10 ** 8),
    # TODO: Compute the task deadline
    min_memory=25 * BYTES_IN_MB,
    max_memory=100 * BYTES_IN_MB,
    min_datasize=51200,  # Each task produces between 50KB and 200KB of data
    max_datasize=204800,
    density_values=[0.4, 0.5, 0.6, 0.7, 0.8],
    regularity_values=[0.2, 0.5, 0.8],
    fat_values=[0.4, 0.5, 0.6, 0.7, 0.8],
    ccr_values=[0.3, 0.4, 0.5],
    jump_values=[1, 2],
)
