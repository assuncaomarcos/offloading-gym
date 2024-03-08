#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from ..cluster import Cluster


class Scheduler(ABC):
    cluster: Cluster

    def __init__(self,
                 num_edge_resources: int,
                 edge_resource_fps: float,
                 num_user_devices: int,
                 user_device_fps: float,
                 comm_link_mbps: int
                 ):
        self.cluster = Cluster(
            num_edge_resources, edge_resource_fps, num_user_devices, user_device_fps, comm_link_mbps
        )

    @abstractmethod
    def schedule_tasks(self, tasks):
        pass