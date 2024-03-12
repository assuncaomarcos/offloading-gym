#!/usr/bin/env python
# -*- coding: utf-8 -*-

from availability.profile import DiscreteProfile, ContinuousProfile

BITS_IN_MEGABIT = 10 ** 6


class Cluster(object):
    edge_resources: DiscreteProfile
    user_devices: DiscreteProfile
    comm_link: ContinuousProfile
    edge_resource_fps: float         # CPU capability in flops per second
    user_device_fps: float
    comm_link_mbps: int              # Link between user devices and edge server in Mbps

    def __init__(self,
                 num_edge_resources: int,
                 edge_resource_fps: float,
                 num_user_devices: int,
                 user_device_fps: float,
                 comm_link_mbps: int):
        self.edge_resource_fps = edge_resource_fps
        self.user_device_fps = user_device_fps
        self.comm_link_mbps = comm_link_mbps
        self.edge_resources = DiscreteProfile(num_edge_resources)
        self.user_devices = DiscreteProfile(num_user_devices)
        self.comm_link = ContinuousProfile(comm_link_mbps)

    @property
    def num_edge_resources(self) -> int:
        return self.edge_resources.max_capacity

    @property
    def num_user_devices(self) -> int:
        return self.user_devices.max_capacity

    def edge_execution_time(self, num_fps: int) -> float:
        """
        Computes the execution time on the edge resources.
        It assumes a task can use all resources and that the parallelization overhead is negligible
        """
        return num_fps / (self.num_edge_resources * self.edge_resource_fps)

    def local_execution_time(self, num_fps: int) -> float:
        """ Computes the execution time of a task on a user device """
        return num_fps / self.user_device_fps

    def transmission_time(self, num_bytes: int) -> float:
        """ Computes the time to transfer a given number of bytes through communication link """
        return (num_bytes * 8 / BITS_IN_MEGABIT) / self.comm_link_mbps

