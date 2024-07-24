#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides a gymnasium environment for evaluating the
placement of task DAGs onto fog resources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Dict, Union, List, Generic, TypeVar
from numpy.typing import NDArray

import gymnasium as gym
import numpy as np
import simpy

from offloading_gym.simulation.fog.resources import ComputingEnvironment, GeolocationResource
from offloading_gym.simulation.fog.config import ComputingConfig, DEFAULT_COMP_CONFIG, DEFAULT_WORKLOAD_CONFIG
from offloading_gym.task_graph import TaskGraph, TaskTuple, TaskAttr
from offloading_gym.envs.workload import FogDAGWorkload

from .base import BaseOffEnv
from .mixins import TaskGraphMixin


TASKS_PER_APPLICATION = 20
MIN_MAX_LATITUDE = (-90, 90)
MIN_MAX_LONGITUDE = (-180, 180)
CYCLES_IN_GHZ = 1000000000


R = TypeVar('R')


@dataclass
class StateEncoder(ABC, Generic[R]):
    high: NDArray[np.float32] = field(init=False)
    low: NDArray[np.float32] = field(init=False)

    @abstractmethod
    def __call__(self, obj: R) -> NDArray[np.float32]:
        ...

    def __post_init__(self):
        self.high = self._provide_high()
        self.low = self._provide_low()

    @abstractmethod
    def _provide_high(self) -> NDArray[np.float32]:
        ...

    @abstractmethod
    def _provide_low(self) -> NDArray[np.float32]:
        ...

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
                max_num_cores,      # max available cores
                max_memory         # max available memory
            ], dtype=np.float32
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
                0.0],  # min available memory
            dtype=np.float32
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
                obj.available_memory
                # queueing delay
            ],
            dtype=np.float32
        )


@dataclass
class TaskEncoder(StateEncoder[TaskAttr]):
    comp_config: ComputingConfig

    def _provide_high(self) -> NDArray[np.float32]:
        return np.array([1.0], dtype=np.float32)

    def _provide_low(self) -> NDArray[np.float32]:
        return np.array([0.0], dtype=np.float32)

    def __call__(self, obj: TaskAttr) -> NDArray[np.float32]:
        return np.array([
            obj.processing_demand
        ], dtype=np.float32)


class FogPlacementEnv(BaseOffEnv, TaskGraphMixin):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Dict
    workload: FogDAGWorkload
    server_encoder: ServerEncoder
    task_encoder: TaskEncoder
    computing_config: Union[ComputingConfig, None]
    computing_env: Union[ComputingEnvironment, None]
    task_graph: Union[TaskGraph, None]  # Current task graph

    # To avoid sorting tasks multiple times
    _task_list: Union[List[TaskTuple], None]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.computing_config = kwargs.get("computing_config", DEFAULT_COMP_CONFIG)
        self.workload_config = kwargs.get("workload_config", DEFAULT_WORKLOAD_CONFIG)
        self.computing_env = None

        self.server_encoder = kwargs.get(
            "server_encoder", ServerEncoder(
                comp_config=self.computing_config
            )
        )
        self.task_encoder = kwargs.get(
            "task_encoder", TaskEncoder(
                comp_config=self.computing_config
            )
        )
        self._setup_spaces()
        self._build_simulation(kwargs)

    def _setup_spaces(self):
        self.action_space = gym.spaces.Discrete(
            n=self.computing_config.num_compute_resources()
        )
        num_servers = self.computing_config.num_compute_resources()
        server_min, server_max = min(self.server_encoder.low), max(self.server_encoder.high)
        task_min, task_max = min(self.task_encoder.low), max(self.task_encoder.high)
        self.observation_space = gym.spaces.Dict(
            {
                "servers": gym.spaces.Box(
                    low=server_min,
                    high=server_max,
                    shape=(num_servers, self.server_encoder.num_features)
                ),
                "task": gym.spaces.Box(
                    low=task_min,
                    high=task_max,
                    shape=(self.task_encoder.num_features,)
                )
            }
        )

        # print(self.observation_space)

    def _build_simulation(self, kwargs):
        self.workload = FogDAGWorkload.build(self.workload_config)

    def _setup_computing_env(self, seed: int) -> ComputingEnvironment:
        simpy_env = simpy.Environment()
        return ComputingEnvironment.build(
            seed=seed,
            simpy_env=simpy_env,
            config=self.computing_config,
        )

    def _compute_task_runtime(self, task: TaskAttr) -> float:
        iot_device = self.computing_env.iot_devices[0]
        return task.processing_demand / (iot_device.cpu_core_speed * CYCLES_IN_GHZ)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, NDArray[np.float32]], dict[str, Any]]:
        super().reset(seed=seed)
        self.workload.reset(seed=seed)

        # Use only the first task graph
        self.task_graph = self.workload.step(offset=0)[0]
        self.compute_task_ranks(self.task_graph, self._compute_task_runtime)

        if not self.computing_env:
            self.computing_env = self._setup_computing_env(seed=seed)

        return self._get_ob(), {}

    def _get_ob(self) -> Dict[str, NDArray[np.float32]]:
        assert self.state is not None, "Call reset before using FogPlacementEnv."
        return self.state

    def step(self, action: int) -> Tuple[
        NDArray[np.float32],
        np.float32,
        bool,
        bool,
        Dict[str, Any],
    ]:
        ...

    def _server_embedding(self) -> NDArray[np.float32]:
        resources = self.computing_env.comp_resources.values()
        server_arrays = [self.server_encoder(resource) for resource in resources]
        return np.stack(server_arrays, axis=0)

    def _task_embedding(self) -> NDArray[np.float32]:
        ...

    @property
    def state(self) -> Dict[str, NDArray[np.float32]]:
        # print(self._server_embedding())
        return {
            "servers": self._server_embedding(),
            "task": np.array([0], dtype=np.float32)
        }
