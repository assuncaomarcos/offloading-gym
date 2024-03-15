#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Generic, List
from ..task_graph import TaskAttr
from ..cluster import Cluster
from gymnasium.core import ActType


class Scheduler(ABC, Generic[ActType]):
    _cluster: Cluster

    def __init__(
            self,
            num_edge_resources: int,
            edge_resource_fps: float,
            num_user_devices: int,
            user_device_fps: float,
            comm_link_mbps: int
    ):
        self._cluster = Cluster(
            num_edge_resources, edge_resource_fps, num_user_devices, user_device_fps, comm_link_mbps
        )

    @property
    def cluster(self):
        return self._cluster

    def reset(self):
        self.cluster.reset()

    @abstractmethod
    def compute_schedule(self, tasks: List[TaskAttr], action: ActType):
        """Computes a schedule of tasks onto the available resources."""
