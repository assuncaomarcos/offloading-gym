#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides a gymnasium environment for evaluating the
placement of task DAGs onto fog resources.
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

from offloading_gym.simulation.fog.resources import (
    ComputingEnvironment,
    GeolocationResource,
)
from offloading_gym.simulation.fog.config import (
    ComputingConfig,
    DEFAULT_COMP_CONFIG,
    DEFAULT_WORKLOAD_CONFIG,
    WorkloadConfig,
)
from offloading_gym.simulation.fog.simulation import (
    FogSimulation,
    TaskRunInfo,
)

from offloading_gym.task_graph import TaskGraph, TaskTuple, TaskAttr
from offloading_gym.envs.workload import FogDAGWorkload, FogTaskAttr
from offloading_gym.utils import arrays

from .base import BaseOffEnv
from .mixins import TaskGraphMixin

FogTaskTuple = Tuple[TaskTuple, FogTaskAttr]

TASKS_PER_APPLICATION = 20
MIN_MAX_LATITUDE = (-90, 90)
MIN_MAX_LONGITUDE = (-180, 180)

# Number of preceding and downstream tasks to consider in the dependency model
NUM_TASK_PREDECESSORS = 6
NUM_TASK_SUCCESSORS = NUM_TASK_PREDECESSORS
NUM_PLACEMENT_TASKS = NUM_TASK_PREDECESSORS


R = TypeVar("R")


@dataclass
class StateEncoder(ABC, Generic[R]):
    high: NDArray[np.float32] = field(init=False)
    low: NDArray[np.float32] = field(init=False)

    @abstractmethod
    def __call__(self, obj: R) -> NDArray[np.float32]: ...

    def __post_init__(self):
        self.high = self._provide_high()
        self.low = self._provide_low()

    @abstractmethod
    def _provide_high(self) -> NDArray[np.float32]: ...

    @abstractmethod
    def _provide_low(self) -> NDArray[np.float32]: ...

    @property
    def num_features(self) -> int:
        """Returns the number of server features considered by this encoder"""
        return self.high.size


@dataclass
class ServerEncoder(StateEncoder[GeolocationResource]):
    """Default class used to encode a server's attributes as a numpy array."""

    comp_config: ComputingConfig

    def _provide_high(self) -> NDArray[np.float32]:
        max_num_cores = self.comp_config.max_number_cores()
        max_memory = self.comp_config.max_memory()
        return np.array(
            [
                self.comp_config.num_compute_resources() - 1,
                max_num_cores,
                self.comp_config.max_core_speed(),
                max_memory,
                MIN_MAX_LATITUDE[1],
                MIN_MAX_LONGITUDE[1],
                max_num_cores,  # max available cores
                max_memory,  # max available memory
            ],
            dtype=np.float32,
        )

    def _provide_low(self) -> NDArray[np.float32]:
        return np.array(
            [
                0.0,  # resource id
                0.0,  # min number of cores
                0.0,  # min core speed
                0.0,  # min memory
                MIN_MAX_LATITUDE[0],
                MIN_MAX_LONGITUDE[0],
                0.0,  # min available cores
                0.0,
            ],  # min available memory
            dtype=np.float32,
        )

    def __call__(self, obj: GeolocationResource) -> NDArray[np.float32]:
        return np.array(
            object=[
                obj.resource_id,
                obj.number_of_cores,
                obj.cpu_core_speed,
                obj.memory_capacity,
                obj.location.lat,
                obj.location.long,
                obj.available_cpu_cores,
                obj.available_memory,
                # queueing delay
            ],
            dtype=np.float32,
        )


@dataclass
class TaskEncoder(StateEncoder[Tuple[TaskGraph, FogTaskAttr]]):
    comp_config: ComputingConfig
    workload_config: WorkloadConfig

    def _provide_high(self) -> NDArray[np.float32]:
        max_num_tasks = max(self.workload_config.num_tasks)
        id_last_resource = self.comp_config.num_compute_resources() - 1
        dependency_highs = np.full(
            NUM_TASK_SUCCESSORS + NUM_TASK_PREDECESSORS, max_num_tasks
        )
        placement_highs = np.full(NUM_PLACEMENT_TASKS, id_last_resource)

        highs = np.array(
            [self.workload_config.max_computing, self.workload_config.max_memory],
            dtype=np.float32,
        )
        return np.concatenate((highs, dependency_highs, placement_highs))

    def _provide_low(self) -> NDArray[np.float32]:
        dependency_lows = np.full(NUM_TASK_SUCCESSORS + NUM_TASK_PREDECESSORS, -1.0)
        placement_lows = np.full(NUM_PLACEMENT_TASKS, -1.0)

        lows = np.array(
            [self.workload_config.min_computing, self.workload_config.min_memory],
            dtype=np.float32,
        )
        return np.concatenate((lows, dependency_lows, placement_lows))

    @classmethod
    def _task_dependencies(cls, task_graph: TaskGraph, task_id: int) -> List[int]:
        task_predecessors = list(task_graph.pred[task_id].keys())
        task_predecessors = arrays.pad_list(
            lst=task_predecessors,
            target_length=NUM_TASK_PREDECESSORS,
            pad_value=-1.0,
        )

        placement_predecessors = [
            task_attr.resource_id for task_attr in task_graph.predecessors(task_id)
        ]
        placement_predecessors = arrays.pad_list(
            lst=placement_predecessors,
            target_length=NUM_PLACEMENT_TASKS,
            pad_value=-1.0,
        )

        task_successors = list(task_graph.succ[task_id].keys())
        task_successors = arrays.pad_list(
            lst=task_successors, target_length=NUM_TASK_SUCCESSORS, pad_value=-1.0
        )

        return task_predecessors + task_successors + placement_predecessors

    def __call__(self, obj: Tuple[TaskGraph, FogTaskAttr]) -> NDArray[np.float32]:
        task_graph, task_attr = obj

        dependency_array = np.array(
            self._task_dependencies(task_graph, task_attr.task_id), dtype=np.float32
        )

        task_embedding = np.array(
            [task_attr.processing_demand, task_attr.memory], dtype=np.float32
        )

        return np.concatenate((task_embedding, dependency_array))


class FogPlacementEnv(BaseOffEnv, TaskGraphMixin):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Dict
    workload: FogDAGWorkload
    server_encoder: ServerEncoder
    task_encoder: TaskEncoder
    compute_config: Union[ComputingConfig, None] = None
    compute_env: Union[ComputingEnvironment, None] = None
    task_graph: Union[TaskGraph, None] = None  # Current task graph

    # To avoid sorting tasks multiple times
    _task_queue: Union[Deque[FogTaskAttr], None] = None
    _simulation: FogSimulation

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compute_config = kwargs.get("compute_config", DEFAULT_COMP_CONFIG)
        self.workload_config = kwargs.get("workload_config", DEFAULT_WORKLOAD_CONFIG)
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
            self.compute_env = self._setup_compute_env(seed=seed)

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
            self.task_graph, key=lambda task_id: self.task_graph.nodes[task_id].rank
        )

        # Store sorted tasks to avoid having to sort it multiple times
        self._task_queue = deque([self.task_graph.nodes[node] for node in topo_order])

        return self._get_ob(), {}

    def _get_ob(self) -> Dict[str, NDArray[np.float32]]:
        assert self.state is not None, "Call reset before using FogPlacementEnv."
        return self.state

    def step(self, action: int) -> Tuple[
        Dict[str, NDArray[np.float32]],
        float,
        bool,
        bool,
        Dict[str, Any],
    ]:
        tasks_execute = self._task_queue.pop()
        task_info = self._simulation.simulate(
            tasks=[tasks_execute], target_resources=[action]
        )

        # TODO: Compute the reward here...

        return self._get_ob(), 0.0, False, False, {}

    def _server_embedding(self) -> NDArray[np.float32]:
        resources = self.compute_env.compute_resources.values()
        server_arrays = [self.server_encoder(resource) for resource in resources]
        return np.stack(server_arrays, axis=0)

    @property
    def state(self) -> Dict[str, NDArray[np.float32]]:
        return {
            "servers": self._server_embedding(),
            "task": self.task_encoder((self.task_graph, self.current_task)),
        }

    @property
    def current_task(self) -> FogTaskAttr:
        """The current task being scheduled/executed"""
        return self._task_queue[0]
