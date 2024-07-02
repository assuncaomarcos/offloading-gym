#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing typing classes for the fog simulation."""

from __future__ import annotations

from typing import List, NamedTuple, Union, AnyStr


class Coordinate(NamedTuple):
    """Namedtuple that represents a geographical coordinate."""

    lat: float
    long: float


class RectGeographicalArea:
    """
    Set of geographical coordinates that form a rectangular area on which
    edge resources are deployed.
    """

    def __init__(
        self,
        *,
        northeast: Coordinate,
        northwest: Coordinate,
        southeast: Coordinate,
        southwest: Coordinate,
    ):
        self._northeast = northeast
        self._northwest = northwest
        self._southeast = southeast
        self._southwest = southwest

        if not self._is_rectangle():
            raise ValueError("Coordinates do not form a rectangle")

    @property
    def northeast(self):
        return self._northeast

    @property
    def northwest(self):
        return self._northwest

    @property
    def southeast(self):
        return self._southeast

    @property
    def southwest(self):
        return self._southwest

    def _is_rectangle(self):
        # Check to see if the coordinates form a rectangle
        return (
            self.northeast.lat == self.northwest.lat
            and self.southeast.lat == self.southwest.lat
            and self.northeast.long == self.southeast.long
            and self.northwest.long == self.southwest.long
        )


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


class ResourceGroupConfig(NamedTuple):
    """
    Data type for the configuration of a group of
    resources (e.g., IoT devices, edge servers, cloud servers)
    """

    num_resources: int
    resource_config: ResourceConfig
    network_config: Union[NetworkConfig, None]
    deployment_area: Union[GeographicalArea, List[CloudSite]]


class ComputingConfig(NamedTuple):
    """Data type for the configuration data of a fog environment."""

    iot: ResourceGroupConfig
    edge: ResourceGroupConfig
    cloud: ResourceGroupConfig


class CloudSite(Coordinate):
    title: AnyStr
    country: AnyStr
    latency: Interval
