#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module containing the classes required for building the configuration of a fog environment.
"""

from __future__ import annotations

from typing import (
    Optional,
    List,
    NamedTuple,
    Union,
    AnyStr,
    Dict,
    Any,
    Tuple,
    Type,
    TYPE_CHECKING,
)
from dataclasses import dataclass, asdict
from enum import Enum

if TYPE_CHECKING:
    from .energy import EnergyModel

__all__ = [
    "Coordinate",
    "RectGeographicalArea",
    "ResourceType",
    "ResourceConfig",
    "Interval",
    "NetworkConfig",
    "GeographicalArea",
    "CloudSite",
    "ResourceGroupConfig",
    "ComputingConfig",
    "WorkloadConfig",
]


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


# For specifying a class and its parameters
ClassWithParams = Tuple[Type[Any], Dict[str, Any]]


class ResourceConfig(NamedTuple):
    """Data type for computing resource configuration."""

    cpu_cores: List[int]
    cpu_core_speed: List[float]
    memory: List[float]
    energy_model: Optional[ClassWithParams] = None


class Interval(NamedTuple):
    """Data type for a value interval."""

    min: float
    max: float


class NetworkConfig(NamedTuple):
    """Data type for computing network configuration."""

    bandwidth: Interval
    # Propagation speed for the communication medium in m/s
    propagation_speed: float


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
        return (
            self.iot.num_resources + self.edge.num_resources + self.cloud.num_resources
        )

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
        return self.max_attribute_value("cpu_cores")

    def max_core_speed(self) -> int:
        """Returns the maximum speed a compute core in the environment has."""
        return self.max_attribute_value("cpu_core_speed")

    def max_memory(self) -> int:
        """Returns the maximum memory size a compute resource in the environment has."""
        return self.max_attribute_value("memory")


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