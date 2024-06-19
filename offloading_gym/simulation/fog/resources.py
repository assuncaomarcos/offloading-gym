#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module that provides resources for simulating task
scheduling in a fog environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict

import math

from simpy.resources.resource import Resource, Request, Release
from simpy.core import BoundClass, Environment, SimTime
from simpy.exceptions import SimPyException


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
            f'ComputeRequest<cpu_cores={self.cpu_cores}, '
            f'memory={self.memory}, '
            f'time={self.time}>'
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
        if event.cpu_cores <= self._available_cpu_cores and event.memory <= self._available_memory:
            self._available_memory -= event.memory
            self._available_cpu_cores -= event.cpu_cores
            self.users.append(event)
            event.usage_since = self._env.now
            event.succeed()
        elif event.cpu_cores > self.number_of_cores or event.memory > self._memory_capacity:
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
            f'ComputeResource<n_cpu_cores={self.capacity}, '
            f'cpu_core_speed={self._cpu_core_speed}, '
            f'available_cpu_cores={self._available_cpu_cores}, '
            f'memory_capacity={self._memory_capacity}, '
            f'available_memory={self._available_memory}, '
            f'queue=[{[str(r) for r in self.queue]}]>'
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
            f'GeolocationResource<resource_id={self.resource_id}, '
            f'latitude={self.latitude}, '
            f'longitude={self.longitude}>, '
            f'n_cpu_cores={self.capacity}, '
            f'cpu_core_speed={self.cpu_core_speed}, '
            f'available_cpu_cores={self.available_cpu_cores}, '
            f'memory_capacity={self.memory_capacity}, '
            f'available_memory={self.available_memory}, '
            f'queue=[{[str(r) for r in self.queue]}]>'
        )

    @staticmethod
    def _to_km(latitude, longitude):
        radius = 6371.0  # approximate radius of earth in km
        latitude = math.radians(latitude)
        longitude = math.radians(longitude)
        return radius * latitude, radius * math.cos(latitude) * longitude

    def manhattan_distance(self, resource: GeolocationResource) -> float:
        lat1, lon1 = self._to_km(self.latitude, self.longitude)
        lat2, lon2 = self._to_km(resource.latitude, resource.longitude)
        return abs(lat1 - lat2) + abs(lon1 - lon2)

    def euclidean_distance(self, resource: GeolocationResource) -> float:
        lat1, lon1 = self._to_km(self.latitude, self.longitude)
        lat2, lon2 = self._to_km(resource.latitude, resource.longitude)
        return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


class FogEnvironment:
    _resources: Dict[int, GeolocationResource]

    @staticmethod
    def build(**config) -> FogEnvironment:
        return None
