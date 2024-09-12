#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module provides classes to encode the state of the Fog environment."""

from .workload import FogTaskAttr
from offloading_gym.task_graph import TaskGraph
from offloading_gym.simulation.fog.resources import GeolocatedResource
from offloading_gym.simulation.fog.typing import ComputingConfig, WorkloadConfig
from offloading_gym.utils import arrays

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numpy.typing import NDArray
from typing import Tuple, List, Generic, TypeVar

import numpy as np

TASKS_PER_APPLICATION = 20
MIN_MAX_LATITUDE = (-90, 90)
MIN_MAX_LONGITUDE = (-180, 180)

# Number of preceding and downstream tasks to consider in the dependency model
NUM_TASK_PREDECESSORS = 6
NUM_TASK_SUCCESSORS = NUM_TASK_PREDECESSORS
NUM_PLACEMENT_TASKS = NUM_TASK_PREDECESSORS


R = TypeVar("R")

__all__ = ["StateEncoder", "ServerEncoder", "TaskEncoder"]


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
class ServerEncoder(StateEncoder[GeolocatedResource]):
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

    def __call__(self, obj: GeolocatedResource) -> NDArray[np.float32]:
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
        pred_ids = list(task_graph.pred[task_id].keys())
        pred_tasks = [task_graph.tasks[node_id] for node_id in pred_ids]

        pred_ids = arrays.pad_list(
            lst=pred_ids,
            target_length=NUM_TASK_PREDECESSORS,
            pad_value=-1.0,
        )

        placement_pred_tasks = [task.resource_id for task in pred_tasks]
        placement_pred_tasks = arrays.pad_list(
            lst=placement_pred_tasks,
            target_length=NUM_PLACEMENT_TASKS,
            pad_value=-1.0,
        )

        task_successors = list(task_graph.succ[task_id].keys())
        task_successors = arrays.pad_list(
            lst=task_successors, target_length=NUM_TASK_SUCCESSORS, pad_value=-1.0
        )

        return pred_ids + task_successors + placement_pred_tasks

    def __call__(self, obj: Tuple[TaskGraph, FogTaskAttr]) -> NDArray[np.float32]:
        task_graph, task_attr = obj

        dependency_array = np.array(
            self._task_dependencies(task_graph, task_attr.task_id), dtype=np.float32
        )

        task_embedding = np.array(
            [task_attr.processing_demand, task_attr.memory], dtype=np.float32
        )

        return np.concatenate((task_embedding, dependency_array))
