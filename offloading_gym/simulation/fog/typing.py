#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing typing classes for the fog simulation."""

from __future__ import annotations

from typing import List, NamedTuple, Union, AnyStr
from dataclasses import dataclass
from enum import Enum


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
            raise ValueError('Set of coordinates is not a rectangular area.')

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


@dataclass(frozen=True)
class CloudSite(Coordinate):
    title: AnyStr
    country: AnyStr
    latency: Interval
