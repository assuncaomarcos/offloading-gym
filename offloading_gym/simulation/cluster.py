#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module contains classes used to represent the computing infrastructure as per the MRLCO/DRLTO papers.

       +-------------+      Upload       +------------+
       |             | ----------------> |            |
       | User Device |                   | Edge Server|
       |             | <---------------- |            |
       +-------------+      Download     +------------+

The infrastructure onto which task graphs are scheduled comprise a user device connected to an
edge server to which tasks may be offloaded. The resources are interconnected by a network link
whose upload (device to edge server) and download (edge server to device) capacities
are separately provided.
"""

from ..task_graph import TaskAttr


BITS_IN_MEGABIT = 10 ** 6


class Cluster:
    """
    Represents a cluster of computational resources comprising an edge server and a user device.
    
    The power model and its constraints are described in the following paper:
     
    Quang Dinh et al., "Offloading in Mobile Edge Computing: Task Allocation and Computational Frequency Scaling",
    IEEE Transactions on Communications, Vol. 65, No. 8, August 2017

    Attributes:
        num_edge_cpus (int): The number of available CPUs in the edge server
        edge_cpu_capacity (float): The capacity of each CPU in the edge server, measured in FLOPS
        num_local_cpus (int): The number of available CPUs in the user's computing device
        local_cpu_capacity (float): The processing capacity of each CPU in the user's device, measured in FLOPS
        upload_rate (float): The data upload rate from the device to the server, measured in Mbps
        download_rate (float): The data download rate from the server to the device, measured in Mbps
        power_tx (float): The power consumption for data transmission over the wireless channel, measured in watts
        power_rx (float): The power consumption for data reception over the wireless channel, measured in watts
        power_rho (float): A constant that depends on the average switched capacitance and the average activity factor
        power_zeta (float) is a constant (usually close to 3)
    """

    num_edge_cpus: int
    edge_cpu_capacity: float
    num_local_cpus: int
    local_cpu_capacity: float
    upload_rate: float
    download_rate: float

    power_tx: float
    power_rx: float
    power_rho: float
    power_zeta: float = 3

    def __init__(
            self,
            *,
            num_edge_cpus: int,
            edge_cpu_capacity: float,
            num_local_cpus: int,
            local_cpu_capacity: float,
            upload_rate: float,
            download_rate: float,
            power_tx: float,
            power_rx: float,
            power_cpu: float
    ):
        self.num_edge_cpus = num_edge_cpus
        self.edge_cpu_capacity = edge_cpu_capacity
        self.num_local_cpus = num_local_cpus
        self.local_cpu_capacity = local_cpu_capacity
        self.upload_rate = upload_rate
        self.download_rate = download_rate
        self.power_tx = power_tx
        self.power_rx = power_rx
        self.power_rho = power_cpu / local_cpu_capacity ** self.power_zeta

    def edge_execution_time(self, num_fps: int) -> float:
        """
        Computes the execution time on the edge resources.
        It assumes a task can use all resources and that the parallelization overhead is negligible
        """
        return num_fps / (self.num_edge_cpus * self.edge_cpu_capacity)

    def local_execution_time(self, num_fps: int) -> float:
        """ Computes the execution time of a task on a user device """
        return num_fps / self.local_cpu_capacity

    def upload_time(self, num_bytes: int) -> float:
        """ Computes the time to transfer a given number of bytes through the upload link """
        return (num_bytes * 8 / BITS_IN_MEGABIT) / self.upload_rate

    def download_time(self, num_bytes: int) -> float:
        """ Computes the time to transfer a given number of bytes through the upload link """
        return (num_bytes * 8 / BITS_IN_MEGABIT) / self.download_rate

    def energy_local_execution(self, task: TaskAttr) -> float:
        """ Computes the energy consumption of a task on a user device """
        runtime = self.local_execution_time(task.processing_demand)
        return self.power_rho * (self.local_cpu_capacity ** self.power_zeta) * runtime

    def energy_offloading(self, task: TaskAttr) -> float:
        """ Computes the energy consumption of a task when offloaded """
        return (
                self.power_tx * self.upload_time(task.task_size) +
                self.power_rx * self.download_time(task.output_datasize)
        )
