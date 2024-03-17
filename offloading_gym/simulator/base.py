#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Generic, List
from ..task_graph import TaskAttr
from offloading_gym.simulator.cluster import Cluster
from gymnasium.core import ActType


class Scheduler(ABC, Generic[ActType]):
    _cluster: Cluster

    def __init__(
            self,
            *,
            num_edge_cpus: int,
            edge_cpu_capacity: float,
            user_cpu_capacity: float,
            upload_rate: float,
            download_rate: float
    ):
        self._cluster = Cluster(
            num_edge_cpus=num_edge_cpus,
            edge_cpu_capacity=edge_cpu_capacity,
            local_cpu_capacity=user_cpu_capacity,
            upload_rate=upload_rate,
            download_rate=download_rate
        )

    @property
    def cluster(self):
        return self._cluster

    @abstractmethod
    def reset(self):
        """Resets any state variables associated with this simulator"""

    @abstractmethod
    def add_tasks(self, tasks: List[TaskAttr]):
        """Adds tasks to the execution queue."""

    @abstractmethod
    def compute_schedule(self, tasks: List[TaskAttr], scheduling_plan: ActType):
        """Computes a schedule of tasks onto the available resources."""
