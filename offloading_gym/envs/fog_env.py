#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides a gymnasium environment for evaluating the
placement of task DAGs onto fog resources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple, Dict, Union, Callable, List
from numpy.typing import NDArray

import gymnasium as gym
import numpy as np

from offloading_gym.simulation.fog.resources import ComputingEnvironment, GeolocationResource
from offloading_gym.simulation.fog.config import ComputingConfig, DEFAULT_COMP_CONFIG
from offloading_gym.task_graph import TaskGraph, TaskTuple

from .base import BaseOffEnv
from .mixins import TaskGraphMixin


TASKS_PER_APPLICATION = 20
MIN_MAX_LATITUDE = (-90, 90)
MIN_MAX_LONGITUDE = (-180, 180)


class ABCServerEncoder(ABC, Callable[[GeolocationResource], NDArray[np.float32]]):
    """
    Abstract class for creating server encoders, used to encode a server's
    attributes as a numpy array.
    """

    @abstractmethod
    def __call__(self, resource: GeolocationResource) -> NDArray[np.float32]:
        ...

    @abstractmethod
    def low_high_values(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Returns the low and high values for each resource feature"""

    @abstractmethod
    def num_features(self) -> int:
        """Returns the number of server features considered by this encoder"""


@dataclass(frozen=True)
class ServerEncoder(ABCServerEncoder):
    """Default class used to encode a server's attributes as a numpy array."""
    comp_config: ComputingConfig
    high: NDArray[np.float32]
    low: NDArray[np.float32]

    def __init__(self, comp_config: ComputingConfig):
        self.comp_config = comp_config
        max_num_cores = comp_config.max_number_cores()
        max_memory = comp_config.max_memory()
        self.high = np.array(
            [
                comp_config.num_compute_resources() - 1,
                max_num_cores,
                comp_config.max_core_speed(),
                max_memory,
                MIN_MAX_LATITUDE[1],
                MIN_MAX_LONGITUDE[1],
                max_num_cores,  # max available cores
                max_memory,     # max available memory
                1.0], dtype=np.float32
        )
        self.low = np.array(
            [
                0.0,  # resource id
                0.0,  # min number of cores
                0.0,  # min core speed
                0.0,  # min memory
                MIN_MAX_LATITUDE[0],
                MIN_MAX_LONGITUDE[0],
                0.0,  # min available cores
                0.0], # min available memory
            dtype=np.float32
        )

    def __call__(self, resource: GeolocationResource) -> NDArray[np.float32]:
        return np.array(
            object=[
                resource.resource_id,
                resource.number_of_cores,
                resource.cpu_core_speed,
                resource.memory_capacity,
                resource.location.lat,
                resource.location.long,
                resource.available_cpu_cores,
                resource.available_memory
                # queueing delay
            ],
            dtype=np.float32
        )

    def low_high_values(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self.low, self.high

    def num_features(self) -> int:
        return self.high.size


class FogPlacementEnv(BaseOffEnv, TaskGraphMixin):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Dict
    _computing_config: Union[ComputingConfig, None]
    _computing_env: Union[ComputingEnvironment, None]
    _tasks_per_app: int
    _server_encoder: ServerEncoder

    # Current task graph
    _task_graph: Union[TaskGraph, None]

    # To avoid having to sort tasks multiple times for scheduling
    _task_list: Union[List[TaskTuple], None]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._computing_config = kwargs.get("computing_config", DEFAULT_COMP_CONFIG)
        self._server_encoder = kwargs.get("server_encoder", ServerEncoder(self._computing_config))
        self._tasks_per_app = kwargs.get("tasks_per_app", TASKS_PER_APPLICATION)
        self._computing_env = None
        self._setup_spaces()

    def _setup_spaces(self):
        self.action_space = gym.spaces.Discrete(
            n=self._computing_config.num_compute_resources()
        )
        num_servers = self._computing_config.num_compute_resources()
        self.observation_space = gym.spaces.Dict(
            {
                "servers": gym.spaces.Box(-1, 1, shape=(num_servers, self._server_encoder.num_features())),
                "task": gym.spaces.Box(-1, 1, shape=(self._tasks_per_app,))
            }
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)


    def step(self, action: int) -> Tuple[
        NDArray[np.float32],
        np.float32,
        bool,
        bool,
        Dict[str, Any],
    ]:
        ...

    @property
    def state(self):
        pass
