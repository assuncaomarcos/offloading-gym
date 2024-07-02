#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that provides resources for simulating task
scheduling in a fog environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Union, Optional

import math
import itertools
import numpy as np
import simpy

from simpy.resources.resource import Resource, Request, Release
from simpy.core import BoundClass, Environment, SimTime
from simpy.exceptions import SimPyException
from gymnasium.utils import seeding

from .backbone import server_info as cloud_sites
from .typing import (
    Coordinate,
    RectGeographicalArea,
    Interval,
    ResourceConfig,
    ResourceGroupConfig,
    ComputingConfig,
    NetworkConfig,
    CloudSite
)


# The following GPS coordinates roughly encompass the entire
# city of Montreal, forming a rough rectangular boundary around it.
MONTREAL_AREA = RectGeographicalArea(
    northeast=Coordinate(lat=45.7057, long=73.4746),
    northwest=Coordinate(lat=45.7057, long=73.9434),
    southeast=Coordinate(lat=45.3831, long=73.4746),
    southwest=Coordinate(lat=45.3831, long=73.9434),
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
        deployment_area=MONTREAL_AREA,
    ),
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
    resource_id: int
    latitude: float
    longitude: float

    def __init__(self, resource_id: int, latitude: float, longitude: float, **kwargs):
        self.resource_id = resource_id
        self.latitude = latitude
        self.longitude = longitude
        super().__init__(**kwargs)

    def __str__(self):
        return (
            f"GeolocationResource<resource_id={self.resource_id}, "
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


class ComputingEnvironment:
    _iot_devices: List[GeolocationResource]
    _edge_resources: List[GeolocationResource]
    _cloud_resources: List[GeolocationResource]
    _all_resources: Dict[int, GeolocationResource]
    _simpy_env: simpy.Environment

    def __init__(
        self,
        simpy_env: simpy.Environment,
        iot_devices: List[GeolocationResource],
        edge_resources: List[GeolocationResource],
        cloud_resources: List[GeolocationResource]
    ):
        self._simpy_env = simpy_env
        self._iot_devices = iot_devices
        self._edge_resources = edge_resources
        self._cloud_resources = cloud_resources
        self._all_resources = {}

        for resource_list in [edge_resources, cloud_resources]:
            for resource in resource_list:
                self._all_resources[resource.resource_id] = resource

    @property
    def simpy_env(self) -> simpy.Environment:
        return self._simpy_env

    @staticmethod
    def build(
        *,
        simpy_env: simpy.Environment,
        seed: Optional[int] = None,
        config: Optional[ComputingConfig] = DEFAULT_COMP_CONFIG,
    ) -> ComputingEnvironment:

        # Initialize the RNG
        rand, seed = seeding.np_random(seed)

        resource_ids = itertools.count(start=0)
        iot_config = config.iot

        # Create the IoT device(s)
        coordinates = ComputingEnvironment.random_coordinates(
            np_random=rand,
            num_locations=iot_config.num_resources,
            area=iot_config.deployment_area,
        )

        iot_resources = ComputingEnvironment.create_resources(
            iot_config, coordinates, resource_ids, rand, simpy_env
        )

        # Create edge resources
        edge_config = config.edge
        coordinates = ComputingEnvironment.grid_coordinates(
            num_locations=edge_config.num_resources, area=edge_config.deployment_area
        )

        edge_resources = ComputingEnvironment.create_resources(
            edge_config, coordinates, resource_ids, rand, simpy_env
        )

        # Create cloud resources
        cloud_config = config.cloud
        coordinates = ComputingEnvironment.dummy_coordinates(
            num_locations=cloud_config.num_resources, area=cloud_config.deployment_area
        )

        cloud_resources = ComputingEnvironment.create_resources(
            cloud_config, coordinates, resource_ids, rand, simpy_env
        )

        return ComputingEnvironment(
            simpy_env=simpy_env,
            iot_devices=iot_resources,
            edge_resources=edge_resources,
            cloud_resources=cloud_resources,
        )

    @staticmethod
    def grid_coordinates(
        num_locations: int, area: RectGeographicalArea
    ) -> List[Coordinate]:
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

    @staticmethod
    def random_coordinates(
        np_random: np.random.Generator, num_locations: int, area: RectGeographicalArea
    ) -> List[Coordinate]:
        lat_values = np_random.uniform(
            area.northwest.lat, area.southeast.lat, size=num_locations
        )
        long_values = np_random.uniform(
            area.northwest.long, area.southeast.long, size=num_locations
        )
        locations = [
            Coordinate(lat=lat, long=long) for lat, long in zip(lat_values, long_values)
        ]
        return locations

    @staticmethod
    def dummy_coordinates(
        num_locations: int, area: RectGeographicalArea
    ) -> List[Coordinate]:
        return [
            Coordinate(lat=area.southeast.lat, long=area.southeast.long)
            for _ in range(num_locations)
        ]

    @staticmethod
    def create_resources(
        config: ResourceGroupConfig,
        coordinates: List[Coordinate],
        resource_ids: itertools.count,
        rand: np.random,
        simpy_env: Environment,
    ):
        return [
            GeolocationResource(
                resource_id=next(resource_ids),
                latitude=coord.lat,
                longitude=coord.long,
                env=simpy_env,
                n_cpu_cores=rand.choice(config.resource_config.cpu_cores),
                cpu_core_speed=rand.choice(config.resource_config.cpu_core_speed),
                memory_capacity=rand.choice(config.resource_config.memory),
            )
            for coord in coordinates
        ]

    @property
    def resources(self) -> List[GeolocationResource]:
        return list(self._all_resources.values())
