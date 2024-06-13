#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from simpy.resources.resource import Resource, Request
from simpy.resources.base import Get
from simpy.core import BoundClass, Environment, SimTime
from simpy.exceptions import SimPyException
from typing import TYPE_CHECKING, List


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


class ComputeRelease(Get):
    """Releases the usage of *resources* granted by *request*. This event is
    triggered immediately. Subclass of :class:`simpy.resources.base.Get`.

    """
    request: ComputeRequest

    def __init__(self, resource: ComputeResource, request: ComputeRequest):
        self.request = request
        """The request (:class:`Request`) that is to be released."""
        super().__init__(resource)


class ComputeResource(Resource):
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
        return self._env

    @property
    def number_of_cores(self) -> int:
        """Return the maximum number of CPU cores in this resource"""
        return self.capacity

    @property
    def cpu_core_speed(self) -> float:
        return self._cpu_core_speed

    @property
    def memory_capacity(self) -> float:
        return self._memory_capacity

    if TYPE_CHECKING:

        def request(self, cpu_cores: int = 1, memory: float = 0.0) -> ComputeRequest:
            """Request a usage slot."""
            return ComputeRequest(self, cpu_cores, memory)

        def release(self, request: ComputeRequest) -> ComputeRelease:
            """Release a usage slot."""
            return ComputeRelease(self, request)

    else:
        request = BoundClass(ComputeRequest)
        release = BoundClass(ComputeRelease)

    def _do_put(self, event: ComputeRequest) -> None:
        if event.cpu_cores <= self._available_cpu_cores and event.memory <= self._available_memory:
            self._available_memory -= event.memory
            self._available_cpu_cores -= event.cpu_cores
            self.users.append(event)
            event.usage_since = self._env.now
            event.succeed()
        elif event.cpu_cores > self.number_of_cores or event.memory > self._memory_capacity:
            event.fail(SimPyException("Request exceeded maximum capacity"))

    def _do_get(self, event: ComputeRelease) -> None:
        try:
            self.users.remove(event.request)  # type: ignore
            self._available_memory += event.request.memory
            self._available_cpu_cores += event.request.cpu_cores
        except ValueError:
            pass
        event.succeed()

    def upload_time(self, req_bytes: int):
        pass

    def execution_time(self, req_cycles: int):
        pass

    def __str__(self):
        return (
            f'ComputeResource<n_cpu_cores={self.capacity}, '
            f'cpu_core_speed={self._cpu_core_speed}, '
            f'available_cpu_cores={self._available_cpu_cores}, '
            f'memory_capacity={self._memory_capacity}, '
            f'available_memory={self._available_memory}, '
            f'queue=[{[str(r) for r in self.queue]}]>'
        )
