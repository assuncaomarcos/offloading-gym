#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Union
from dataclasses import dataclass
from gymnasium.utils import seeding
from typing import Optional, Callable, Dict
from .typing import ComputingConfig, ResourceType
from functools import cache

import simpy
import numpy as np
import copy
import itertools

from offloading_gym.envs.fog.workload import FogTaskAttr

from .resources import GeolocatedResource, NetworkResource

from .typing import (
    NetworkConfig,
    Coordinate,
    RectGeographicalArea,
    GeographicalArea,
    ResourceType,
    ResourceGroupConfig,
    ComputingConfig,
    CloudSite,
)

IOT_DEVICE_ID = 0
GB_IN_BYTES = 1024**3
CYCLES_IN_GHZ = 1_000_000_000
GIGABIT_IN_BYTES = 1_000_000_000 // 8


class ResourceManager:
    """
    Creates the required resources for a discrete event
    simulation based on the environment configuration.
    """

    _simpy_env: simpy.Environment
    _np_random: np.random.Generator
    _config: ComputingConfig

    # Each resource group (iot, edge, cloud) has a specific function for
    # creating/returning the resource geolocations
    _coord_fn: Dict[ResourceType, Callable[[GeographicalArea, int], List[Coordinate]]]

    _resource_ids: itertools.count
    _comp_resources: Dict[int, GeolocatedResource]

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
            ResourceType.EDGE: self._grid_coordinates,
            ResourceType.CLOUD: self._cloud_coordinates,
            ResourceType.IOT: self._random_coordinates,
        }
        self._resource_ids = itertools.count(start=0)
        self._comp_resources = {}

    def initialize(self) -> None:
        """Initializes the resource manager."""
        self._setup_comp_resources()

    @property
    def computing_config(self) -> ComputingConfig:
        """Returns the computing configuration"""
        return self._config

    def compute_resources(self) -> Dict[int, GeolocatedResource]:
        """Returns the computing resources"""
        if not self._comp_resources:
            raise ResourceWarning("Initialize the resource manager first")

        self._setup_comp_resources()
        return self._comp_resources

    def _setup_comp_resources(self):
        """
        Returns the compute resources required for the simulation.

        Returns:
            A dictionary with all the compute resources
        """
        if not self._comp_resources:
            for res_type in [ResourceType.IOT, ResourceType.EDGE, ResourceType.CLOUD]:
                resources = self._create_resource_group(res_type)

                for res in resources:
                    self._comp_resources[res.resource_id] = res

        return self._comp_resources

    @staticmethod
    def _grid_coordinates(
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

    def _random_coordinates(
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
            area.southeast.lat, area.northwest.lat, size=num_locations
        )
        long_values = self._np_random.uniform(
            area.northwest.long, area.southeast.long, size=num_locations
        )
        locations = [
            Coordinate(lat=lat, long=long) for lat, long in zip(lat_values, long_values)
        ]
        return locations

    def _cloud_coordinates(
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

    def _create_resource_group(
        self, resource_type: ResourceType
    ) -> List[GeolocatedResource]:
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
        resources = []
        energy_model_class, energy_model_params = res_config.energy_model
        for coord in coordinates:
            res = GeolocatedResource(
                resource_id=next(self._resource_ids),
                res_type=resource_type,
                location=coord,
                energy_model=energy_model_class(**energy_model_params),
                env=self._simpy_env,
                n_cpu_cores=self._np_random.choice(res_config.cpu_cores),
                cpu_core_speed=self._np_random.choice(res_config.cpu_core_speed),
                memory_capacity=self._np_random.choice(res_config.memory),
            )
            resources.append(res)
        return resources

    @cache
    def network_resource(self, source_id: int, destination_id: int) -> NetworkResource:
        """
        Returns a network resource that represents the
        bandwidth and latency between two compute resources.

        Args:
            source_id: the source resource's id
            destination_id: the destination resource's id'

        Returns:
            A network resource.
        """
        source = self._comp_resources[source_id]
        destination = self._comp_resources[destination_id]
        net_config = self._network_config(destination.resource_type)

        if net_config is None:
            bandwidth = float("inf")
            latency = 0
        else:
            bandwidth = self._np_random.uniform(
                net_config.bandwidth.min, net_config.bandwidth.max
            )
            if destination.resource_type == ResourceType.EDGE:
                distance_km = source.euclidean_distance(destination)
                latency = (distance_km * 1000) / net_config.propagation_speed
            else:
                proxy_latency = destination.location.latency
                latency = self._np_random.uniform(proxy_latency.min, proxy_latency.max)

        return NetworkResource(bandwidth=bandwidth, latency=latency)

    def _network_config(self, dest_type: ResourceType) -> NetworkConfig:
        if dest_type == ResourceType.EDGE:
            return self._config.edge.network_config
        elif dest_type == ResourceType.CLOUD:
            return self._config.cloud.network_config
        else:
            return self._config.iot.network_config


@dataclass(frozen=True)
class ComputingEnvironment:
    """
    This class represents the computing infrastructure to use
    for a given discrete event simulation.latencies: int =
    """

    simpy_env: simpy.Environment
    resource_mgmt: ResourceManager
    np_random: np.random.Generator

    @staticmethod
    def build(
        *,
        seed: Optional[int] = None,
        simpy_env: Optional[simpy.Environment] = None,
        config: Optional[ComputingConfig] = None,
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

        if simpy_env is None:
            simpy_env = simpy.Environment()

        res_mgmt = ResourceManager(simpy_env=simpy_env, np_random=rand, config=config)
        res_mgmt.initialize()
        return ComputingEnvironment(
            simpy_env=simpy_env, resource_mgmt=res_mgmt, np_random=rand
        )

    @property
    def compute_resources(self):
        return self.resource_mgmt.compute_resources()

    def network_link(self, source_id: int, destination_id: int) -> NetworkResource:
        return self.resource_mgmt.network_resource(source_id, destination_id)

    @classmethod
    def task_runtime(cls, resource: GeolocatedResource, task: FogTaskAttr) -> float:
        return task.processing_demand / (resource.cpu_core_speed * CYCLES_IN_GHZ)

    def data_transfer_time(
        self,
        source: GeolocatedResource,
        destination: GeolocatedResource,
        num_bytes: int,
    ):
        net_link = self.network_link(
            source_id=source.resource_id, destination_id=destination.resource_id
        )
        return num_bytes / (net_link.bandwidth * GIGABIT_IN_BYTES) + net_link.latency

    def clone(self) -> ComputingEnvironment:
        """Returns a deep copy of the computing environment."""
        return copy.deepcopy(self)


class FogSimulation:
    sim_env: simpy.Environment
    comp_env: ComputingEnvironment
    simulation_process: Union[simpy.Event, None]
    done_tasks: List[FogTaskAttr]
    iot_device: GeolocatedResource

    def __init__(self, comp_env: ComputingEnvironment):
        self.comp_env = comp_env
        self.sim_env = comp_env.simpy_env
        self.simulation_process = None
        self.iot_device = comp_env.compute_resources.get(IOT_DEVICE_ID)
        self.done_tasks = []

    def _execute_task(self, task: FogTaskAttr, resource: GeolocatedResource):
        task.arrival_time = self.sim_env.now
        with resource.request(cpu_cores=1, memory=task.memory / GB_IN_BYTES) as req:
            yield req
            task.start_time = self.sim_env.now
            task.resource_id = resource.resource_id

            task.upload_delay = self.comp_env.data_transfer_time(
                source=self.iot_device,
                destination=resource,
                num_bytes=task.task_size,
            )
            task.download_delay = self.comp_env.data_transfer_time(
                source=resource,
                destination=self.iot_device,
                num_bytes=task.output_datasize,
            )
            task_runtime = self.comp_env.task_runtime(resource, task)

            yield self.sim_env.timeout(task.upload_delay)
            yield self.sim_env.timeout(task_runtime)
            yield self.sim_env.timeout(task.download_delay)

            task.finish_time = self.sim_env.now
            task.energy_used = resource.energy_use(task=task)

            self.done_tasks.append(task)

    def _task_manager(self, tasks: List[FogTaskAttr], target_resources: List[int]):
        for task_attr, resource_id in zip(tasks, target_resources):
            resource = self.comp_env.compute_resources[resource_id]
            print(f"Managing task {task_attr} on resource {resource}")
            yield self.sim_env.process(self._execute_task(task_attr, resource))

    def simulate(
        self, tasks: List[FogTaskAttr], target_resources: List[int]
    ) -> List[FogTaskAttr]:
        print("simulate")
        sim_env = self.comp_env.simpy_env
       
        
       # tasks = [task.np() for task in tasks]  
        # target_resources = [resource.np() for resource in target_resources]  

      
        self.simulation_process = sim_env.process(
            self._task_manager(tasks, target_resources)
        )

        try:
            print("try")
            sim_env.run(until=self.simulation_process)
            print("check completed")
             
        except simpy.Interrupt as interrupt:
            print(f"Simulation interrupted: {interrupt.cause}")
        print(" self.done_tasks",  self.done_tasks)
        return self.done_tasks
    
    
    @staticmethod
    def build(comp_env: ComputingEnvironment):
        return FogSimulation(comp_env)