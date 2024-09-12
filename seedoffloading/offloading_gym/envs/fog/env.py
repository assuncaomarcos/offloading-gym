#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides a gymnasium environment that simulates a fog 
infrastructure comprised of mobile device, edge servers and cloud 
servers. The application is structured as a DAG whose vertices are 
tasks and edges represente data interdependencies between tasks.

The environment simulates the placement and execution of tasks onto
the available resources. An action consists of selecting a resource
onto which the current action will be offloaded.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Dict, Union, List, Generic, TypeVar, Deque
from numpy.typing import NDArray
from collections import deque
from functools import partial

import gymnasium as gym
import numpy as np
import simpy
import networkx as nx
import math

from offloading_gym.simulation.fog import ComputingEnvironment, FogSimulation
from offloading_gym.simulation.fog.typing import (
    ComputingConfig,
    WorkloadConfig,
    Coordinate,
    RectGeographicalArea,
    ResourceConfig,
    ResourceGroupConfig,
    NetworkConfig,
    Interval,
)

from offloading_gym.task_graph import TaskGraph, TaskTuple

from ..base import BaseOffEnv
from ..mixins import TaskGraphMixin
from .workload import FogDAGWorkload, FogTaskAttr
from .cloud import cloud_sites
from .encoders import ServerEncoder, TaskEncoder
from ...simulation.fog.energy import JingEdgeEnergyModel, JingIoTEnergyModel

FogTaskTuple = Tuple[TaskTuple, FogTaskAttr]


BYTES_PER_MEGABYTE = 2**20

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
            cpu_cores=[1],
            cpu_core_speed=[1.0],
            memory=[1.0],
            energy_model=(
                JingIoTEnergyModel,
                {"energy_coefficient": math.pow(10, -27)},
            ),
        ),
        network_config=None,
        deployment_area=MONTREAL_AREA,
    ),
    edge=ResourceGroupConfig(
        num_resources=36,
        resource_config=ResourceConfig(
            cpu_cores=[4],
            cpu_core_speed=[1.5, 1.8, 2.0],
            memory=[1.0, 2.0, 4.0],
            energy_model=(
                JingEdgeEnergyModel,
                {"energy_per_unit": 9 * math.pow(10, -5)},
            ),
        ),
        network_config=NetworkConfig(
            bandwidth=Interval(min=10, max=12), propagation_speed=3 * 10**8
        ),
        deployment_area=MONTREAL_AREA,
    ),
    cloud=ResourceGroupConfig(
        num_resources=20,
        resource_config=ResourceConfig(
            cpu_cores=[8],
            cpu_core_speed=[2.0, 2.6, 3.0],
            memory=[16.0, 24.0, 32.0],
            energy_model=(
                JingEdgeEnergyModel,
                {"energy_per_unit": 9 * math.pow(10, -5)},
            ),
        ),
        network_config=NetworkConfig(
            bandwidth=Interval(min=4, max=8), propagation_speed=2.07 * 10**8
        ),
        deployment_area=cloud_sites(),
    ),
)

DEFAULT_WORKLOAD_CONFIG = WorkloadConfig(
    num_tasks=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    min_computing=(10**7),
    max_computing=(3 * 10**8),
    min_memory=25 * BYTES_PER_MEGABYTE,
    max_memory=100 * BYTES_PER_MEGABYTE,
    # Each task produces between 50KB and 200KB of data
    min_datasize=51200,
    max_datasize=204800,
    density_values=[0.4, 0.5, 0.6, 0.7, 0.8],
    regularity_values=[0.2, 0.5, 0.8],
    fat_values=[0.4, 0.5, 0.6, 0.7, 0.8],
    ccr_values=[0.3, 0.4, 0.5],
    jump_values=[1, 2],
)


class FogPlacementEnv(BaseOffEnv, TaskGraphMixin):
    """
    Implementation of a fog placement environment for task offloading.
    """

    # An action represents the ID of the resource onto which the current
    # task being scheduled must execute
    action_space: gym.spaces.Discrete

    # The observation space is a dictionary with two keys: "servers" and "task"
    observation_space: gym.spaces.Dict
    workload: FogDAGWorkload

    # The state of the environment is represented as numpy arrays
    # using a server and a task encoder.
    server_encoder: ServerEncoder
    task_encoder: TaskEncoder

    # The configuration of the environment and the computing environment
    # used by the discrete event simulation
    compute_config: Union[ComputingConfig, None]
    compute_env: Union[ComputingEnvironment, None]
    _simulation: FogSimulation

    # The task graph produced by the workload generator
    task_graph: Union[TaskGraph, None]
    critical_path: List[FogTaskAttr]

    # To avoid sorting tasks multiple times
    _task_queue: Union[Deque[FogTaskAttr], None]
    _next_task: Union[FogTaskAttr, None]

    # The weights of latency and energy in the cost function
    _energy_weight: float
    _latency_weight: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compute_config = kwargs.get("compute_config", DEFAULT_COMP_CONFIG)
        self.workload_config = kwargs.get("workload_config", DEFAULT_WORKLOAD_CONFIG)
        self.compute_env = None

        self._latency_weight = kwargs.get("latency_weight", 0.5)
        self._energy_weight = kwargs.get("energy_weight", 1.0 - self._latency_weight)
        assert math.isclose(
            self._latency_weight + self._energy_weight, 1.0
        ), "The sum of cost and latency weights must be 1.0"

        self.server_encoder = ServerEncoder(comp_config=self.compute_config)
        self.task_encoder = TaskEncoder(
            comp_config=self.compute_config, workload_config=self.workload_config
        )
        

        self.workload = FogDAGWorkload.build(self.workload_config)
        self._setup_spaces()

    def _setup_spaces(self):
        self.action_space = gym.spaces.Discrete(
            n=self.compute_config.num_compute_resources()
        )
        num_servers = self.compute_config.num_compute_resources()
        server_min, server_max = min(self.server_encoder.low), max(
            self.server_encoder.high
        )
        task_min, task_max = min(self.task_encoder.low), max(self.task_encoder.high)
        self.observation_space = gym.spaces.Dict(
            {
                "servers": gym.spaces.Box(
                    low=server_min,
                    high=server_max,
                    shape=(num_servers, self.server_encoder.num_features),
                ),
                "task": gym.spaces.Box(
                    low=task_min, high=task_max, shape=(self.task_encoder.num_features,)
                ),
            }
        )

    def _setup_compute_env(self, seed: int) -> ComputingEnvironment:
        simpy_env = simpy.Environment()
        return ComputingEnvironment.build(
            seed=seed,
            simpy_env=simpy_env,
            config=self.compute_config,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, NDArray[np.float32]], dict[str, Any]]:
        super().reset(seed=seed)
        self.workload.reset(seed=seed)

        if not self.compute_env:
            random_seed = self.np_random.integers(low=0, high=2**32)
            self.compute_env = self._setup_compute_env(seed=int(random_seed))

        # First resource should always be the IoT device
        runtime_on_iot = partial(
            self.compute_env.task_runtime, self.compute_env.compute_resources.get(0)
        )

        self._simulation = FogSimulation.build(self.compute_env)

        # Use only the first task graph in this version
        self.task_graph = self.workload.step(offset=0)[0]
        self.compute_task_ranks(self.task_graph, runtime_on_iot)

        # Sorts tasks following their dependencies and ranks
        topo_order = nx.lexicographical_topological_sort(
            self.task_graph, key=lambda task_id: -self.task_graph.nodes[task_id].rank
        )

        # Store the application DAG's critical path for later use
        self.critical_path = nx.dag_longest_path(self.task_graph, weight="rank")

        # Store sorted tasks to avoid having to sort it multiple times
        self._task_queue = deque([self.task_graph.nodes[node] for node in topo_order])
        self._next_task = self._task_queue.popleft()

        return self._get_ob(), {}

    def _get_ob(self) -> Dict[str, NDArray[np.float32]]:
        assert self._next_task, "Call reset before using FogPlacementEnv."
        return self.state

    def step(self, action: int) -> Tuple[
        Dict[str, NDArray[np.float32]],
        float,
        bool,
        bool,
        Dict[str, Any],
    ]:  
    
        print("here is the step")
        completed_tasks = self._simulation.simulate(
            tasks=[self._next_task], target_resources=[action]
        )
        print("whats completed_tasks", completed_tasks)
        
        # tasks_execute = self._task_queue.pop()
        # task_info = self._simulation.simulate(tasks=[tasks_execute], target_resources=[action])
        # print("whats info", task_info)

        cost = self._compute_cost(tasks=completed_tasks)

        self._next_task = self._task_queue.popleft()
        observation = self._get_ob()

        truncated = False
        if len(self._task_queue) == 0:
            self._next_task = None
            truncated = True

        return observation, -cost, truncated, False, {}

    def _compute_cost(self, tasks: List[FogTaskAttr]) -> float:
        cost = 0
        max_latency = 0
        max_energy = 0

        for task in tasks:
            max_latency = max(max_latency, task.makespan)
            max_energy = max(max_energy, task.energy_used)

        for task in tasks:
            in_critical_path = 1 if task in self.critical_path else 0

            cost += (self._latency_weight * (task.makespan / max_latency)) + (
                self._energy_weight * (task.energy_used / max_energy)
            ) * in_critical_path

        return cost

    def _server_embedding(self) -> NDArray[np.float32]:
        resources = self.compute_env.compute_resources.values()
        server_arrays = [self.server_encoder(resource) for resource in resources]
        return np.stack(server_arrays, axis=0)

    @property
    def state(self) -> Dict[str, NDArray[np.float32]]:
        return {
            "servers": self._server_embedding(),
            "task": self.task_encoder((self.task_graph, self._next_task)),
        }