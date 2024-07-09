#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that provides resources for simulating task
scheduling in a fog environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Callable, Optional
from dataclasses import dataclass

import math
import itertools
import numpy as np
import simpy

from simpy.resources.resource import Resource, Request, Release
from simpy.core import BoundClass, Environment, SimTime
from simpy.exceptions import SimPyException
from gymnasium.utils import seeding

from .config import (
    Coordinate,
    RectGeographicalArea,
    GeographicalArea,
    Interval,
    ResourceType,
    ResourceConfig,
    ResourceGroupConfig,
    ComputingConfig,
    NetworkConfig,
    CloudSite,
    DEFAULT_COMP_CONFIG
)


class ComputeRequest(Request):
    """Request usage of the *resource*. The event is triggered once access is
    granted. Subclass of :class:`simpy.resources.base.Put`.

    If the maximum capacity of CPU and memory has not yet been reached, the request is
    triggered immediately. If the maximum number of CPUs or memory has been reached,
    the request is triggered once an earlier request releases its allocated resources.

    The request releases the resources automatically when created within
    a :keyword:`with` statement.

    """

    cpu_cores: int
    memory: float
    time: SimTime

    def __init__(self, resource: ComputeResource, cpu_cores: int, memory: float):
        self.cpu_cores = cpu_cores
        self.memory = memory
        self.time = resource.env.now
        super().__init__(resource)

    def __str__(self):
        return (
            f"ComputeRequest<cpu_cores={self.cpu_cores}, "
            f"memory={self.memory}, "
            f"time={self.time}>"
        )


class ComputeResource(Resource):
    """A compute resource used in the fog environment.
    It can represent a mobile device, an edge server or a cloud server"""

    _cpu_core_speed: float
    """Capacity of each CPU core in GHz/second"""
    _memory_capacity: float
    """Overall memory capacity in GB"""
    _available_cpu_cores: int
    """Number of available CPU cores in this resource"""
    _available_memory: float
    """Available memory in GB"""

    PutQueue = list
    GetQueue = list

    def __init__(
        self,
        env: Environment,
        n_cpu_cores: int = 1,
        cpu_core_speed: float = 1,
        memory_capacity: float = 1,
    ):
        super().__init__(env=env, capacity=n_cpu_cores)
        self._cpu_core_speed = cpu_core_speed
        self._available_cpu_cores = n_cpu_cores
        self._memory_capacity = self._available_memory = memory_capacity
        self.users: List[ComputeRequest] = []

    @property
    def env(self):
        """Return the environment associated with the resource"""
        return self._env

    @property
    def number_of_cores(self) -> int:
        """Return the maximum number of CPU cores in this resource"""
        return self.capacity

    @property
    def cpu_core_speed(self) -> float:
        """Return the CPU core speed in GHz/second"""
        return self._cpu_core_speed

    @property
    def memory_capacity(self) -> float:
        """Return the memory capacity in GB"""
        return self._memory_capacity

    @property
    def available_cpu_cores(self) -> int:
        """Return the available CPU cores in this resource"""
        return self._available_cpu_cores

    @property
    def available_memory(self) -> float:
        """Return the available memory in GB"""
        return self._available_memory

    if TYPE_CHECKING:

        def request(self, cpu_cores: int = 1, memory: float = 0.0) -> ComputeRequest:
            """Request a usage slot."""
            return ComputeRequest(self, cpu_cores, memory)

        def release(self, request: ComputeRequest) -> Release:
            """Release a usage slot."""
            return Release(self, request)

    else:
        request = BoundClass(ComputeRequest)
        release = BoundClass(Release)

    def _do_put(self, event: ComputeRequest) -> None:
        if (
            event.cpu_cores <= self._available_cpu_cores
            and event.memory <= self._available_memory
        ):
            self._available_memory -= event.memory
            self._available_cpu_cores -= event.cpu_cores
            self.users.append(event)
            event.usage_since = self._env.now
            event.succeed()
        elif (
            event.cpu_cores > self.number_of_cores
            or event.memory > self._memory_capacity
        ):
            event.fail(SimPyException("Request exceeded maximum capacity"))

    def _do_get(self, event: Release) -> None:
        try:
            self.users.remove(event.request)  # type: ignore
            if isinstance(event.request, ComputeRequest):
                self._available_memory += event.request.memory
                self._available_cpu_cores += event.request.cpu_cores
        except ValueError:
            pass
        event.succeed()

    def __str__(self):
        return (
            f"ComputeResource<n_cpu_cores={self.capacity}, "
            f"cpu_core_speed={self._cpu_core_speed}, "
            f"available_cpu_cores={self._available_cpu_cores}, "
            f"memory_capacity={self._memory_capacity}, "
            f"available_memory={self._available_memory}, "
            f"queue=[{[str(r) for r in self.queue]}]>"
        )


class GeolocationResource(ComputeResource):
    """
    It represents a computing resource with geolocation information.
    """
    resource_id: int
    latitude: float
    longitude: float
    resource_type: ResourceType

    def __init__(
        self,
        resource_id: int,
        res_type: ResourceType,
        latitude: float,
        longitude: float,
        **kwargs,
    ):
        self.resource_id = resource_id
        self.resource_type = res_type
        self.latitude = latitude
        self.longitude = longitude
        super().__init__(**kwargs)

    def __str__(self):
        return (
            f"GeolocationResource<resource_id={self.resource_id}, "
            f"resource_type={self.resource_type}, "
            f"latitude={self.latitude}, "
            f"longitude={self.longitude}>, "
            f"n_cpu_cores={self.capacity}, "
            f"cpu_core_speed={self.cpu_core_speed}, "
            f"available_cpu_cores={self.available_cpu_cores}, "
            f"memory_capacity={self.memory_capacity}, "
            f"available_memory={self.available_memory}, "
            f"queue=[{[str(r) for r in self.queue]}]>"
        )

    @staticmethod
    def _to_km(latitude, longitude):
        radius = 6371.0  # approximate radius of earth in km
        latitude = math.radians(latitude)
        longitude = math.radians(longitude)
        return radius * latitude, radius * math.cos(latitude) * longitude

    def manhattan_distance(self, resource: GeolocationResource) -> float:
        """Computes the Manhattan distance between GeolocationResources."""
        lat1, lon1 = self._to_km(self.latitude, self.longitude)
        lat2, lon2 = self._to_km(resource.latitude, resource.longitude)
        return abs(lat1 - lat2) + abs(lon1 - lon2)

    def euclidean_distance(self, resource: GeolocationResource) -> float:
        """Computes the Euclidean distance between GeolocationResources."""
        lat1, lon1 = self._to_km(self.latitude, self.longitude)
        lat2, lon2 = self._to_km(resource.latitude, resource.longitude)
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


class ResourceManager:
    """
    Creates the required resources for a discrete event
    simulation based on the environment configuration.
    """
    _simpy_env: simpy.Environment
    _np_random: np.random.Generator
    _config: ComputingConfig
    _coord_fn: Dict[ResourceType, Callable[[GeographicalArea, int], List[Coordinate]]]
    _resource_ids: itertools.count

    def __init__(
        self,
        simpy_env: simpy.Environment,
        np_random: np.random.Generator,
        config: ComputingConfig,
    ):
        self._simpy_env = simpy_env
        self._np_random = np_random
        self._config = config
        self._coord_fn = {
            ResourceType.EDGE: self.grid_coordinates,
            ResourceType.CLOUD: self.cloud_coordinates,
            ResourceType.IOT: self.random_coordinates,
        }
        self._resource_ids = itertools.count(start=0)

    @staticmethod
    def grid_coordinates(
        area: RectGeographicalArea, num_locations: int
    ) -> List[Coordinate]:
        """
        Creates a grid in the provided rectangular geographical area
        and computes the geolocation of resources considering one
        resource per area in the grid.

        Args:
            area: a rectangular geographical area.
            num_locations: the number of locations in the grid.

        Returns:
            A list of coordinates.
        """
        locations_per_side = round(np.sqrt(num_locations))
        lat_values = np.linspace(
            area.northwest.lat, area.southeast.lat, num=locations_per_side
        )
        long_values = np.linspace(
            area.northwest.long, area.southeast.long, num=locations_per_side
        )
        locations = [
            Coordinate(lat=lat, long=long) for lat in lat_values for long in long_values
        ]

        return locations[:num_locations]

    def random_coordinates(
        self, area: RectGeographicalArea, num_locations: int
    ) -> List[Coordinate]:
        """
        Randomly selects geolocations that fall into the provided
        rectangular geographical area.

        Args:
            area: a rectangular geographical area.
            num_locations: the number of locations.

        Returns:
            A list of coordinates.
        """
        lat_values = self._np_random.uniform(
           area.southeast.lat,  area.northwest.lat, size=num_locations
        )
        long_values = self._np_random.uniform(
            area.northwest.long, area.southeast.long, size=num_locations
        )
        locations = [
            Coordinate(lat=lat, long=long) for lat, long in zip(lat_values, long_values)
        ]
        return locations

    def cloud_coordinates(
        self, sites: List[CloudSite], num_locations: int
    ) -> List[CloudSite]:
        """
        Randomly selects geolocations from the list of servers
        provided by the Wonderproxy dataset.

        Args:
            sites: the list of servers.
            num_locations: the required number of locations.

        Returns:
            A list of cloud sites.
        """
        indices = self._np_random.choice(len(sites), num_locations)
        return [sites[i] for i in indices]

    def create_resources(self) -> List[GeolocationResource]:
        """Creates the required resources for a discrete event simulation."""
        resources = []
        for res_type in [ResourceType.IOT, ResourceType.EDGE, ResourceType.CLOUD]:
            resources += self.create_resource_group(res_type)
        return resources

    def create_resource_group(self, resource_type: ResourceType):
        """
        Creates the required resources of a given type.

        Args:
            resource_type: the type of resource to create (cloud, edge, iot).

        Returns:
            The list of resources.
        """
        config: ResourceGroupConfig = getattr(self._config, f"{resource_type}")
        get_coordinates = self._coord_fn[resource_type]
        coordinates = get_coordinates(config.deployment_area, config.num_resources)
        res_config = config.resource_config
        return [
            GeolocationResource(
                resource_id=next(self._resource_ids),
                res_type=resource_type,
                latitude=coord.lat,
                longitude=coord.long,
                env=self._simpy_env,
                n_cpu_cores=self._np_random.choice(res_config.cpu_cores),
                cpu_core_speed=self._np_random.choice(res_config.cpu_core_speed),
                memory_capacity=self._np_random.choice(res_config.memory),
            )
            for coord in coordinates
        ]


@dataclass(frozen=True)
class ComputingEnvironment:
    """
    This class represents the computing infrastructure to use
    for a given discrete event simulation.
    """
    simpy_env: simpy.Environment
    resources: Dict[int, GeolocationResource]

    @staticmethod
    def build(
        *,
        simpy_env: simpy.Environment,
        seed: Optional[int] = None,
        config: Optional[ComputingConfig] = DEFAULT_COMP_CONFIG,
    ) -> ComputingEnvironment:
        """
        Builds a computing environment for discrete event simulations.

        Args:
            simpy_env: the simpy environment to use
            seed: the RNG seed for reproducibility
            config: the environment configuration

        Returns:
            A computing environment
        """

        # Initialize the RNG
        rand, seed = seeding.np_random(seed)
        res_mgmt = ResourceManager(simpy_env=simpy_env, np_random=rand, config=config)

        resources = res_mgmt.create_resources()
        resource_dict = {resource.resource_id: resource for resource in resources}

        return ComputingEnvironment(simpy_env=simpy_env, resources=resource_dict)
