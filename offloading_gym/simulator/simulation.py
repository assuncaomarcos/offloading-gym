#!/usr/bin/env python
# -*- coding: utf-8 -*-

import simpy
from .cluster import Cluster
from ..task_graph import TaskAttr, TaskTuple
from typing import List, Union
from enum import IntEnum
from dataclasses import dataclass


class ExecutionType(IntEnum):
    LOCAL = 0
    EDGE = 1


@dataclass
class TaskExecution:
    task_id: int
    finish_time: float
    make_span: float
    energy: float
    execution_type: ExecutionType


class Simulator:
    cluster: Cluster
    sim_env: simpy.Environment
    local_device: simpy.Resource
    edge_server: simpy.Resource
    task_info: List[TaskExecution]
    running_simulation_process: Union[simpy.Event, None]

    def __init__(self, cluster: Cluster):
        self.sim_env = simpy.Environment()
        self.cluster = cluster

        self.edge_server = simpy.Resource(self.sim_env, capacity=self.cluster.num_edge_cpus)
        self.local_device = simpy.Resource(self.sim_env, capacity=self.cluster.num_local_cpus)
        self.task_info = []
        self.running_simulation_process = None

    def upload(self, data_size):
        yield self.sim_env.timeout(self.cluster.upload_time(data_size))

    def download(self, data_size):
        yield self.sim_env.timeout(self.cluster.download_time(data_size))

    def execute_local(self, task: TaskAttr):
        with self.local_device.request() as req:
            yield req
            start_time = self.sim_env.now
            yield self.sim_env.timeout(self.cluster.local_execution_time(task.processing_demand))
            finish_time = self.sim_env.now
            self.task_info.append(
                TaskExecution(
                    task_id=task.task_id,
                    finish_time=finish_time,
                    make_span=finish_time - start_time,
                    energy=self.cluster.energy_local_execution(task),
                    execution_type=ExecutionType.LOCAL
                )
            )

    def execute_remote(self, task: TaskAttr):
        with self.edge_server.request() as req:
            yield req
            start_time = self.sim_env.now
            yield self.sim_env.process(self.upload(task.task_size))
            yield self.sim_env.timeout(self.cluster.edge_execution_time(task.processing_demand))
            yield self.sim_env.process(self.download(task.output_datasize))
            finish_time = self.sim_env.now
            self.task_info.append(
                TaskExecution(
                    task_id=task.task_id,
                    finish_time=finish_time,
                    make_span=finish_time - start_time,
                    energy=self.cluster.energy_offloading(task),
                    execution_type=ExecutionType.EDGE
                )
            )

    def task_manager(self, tasks: List[TaskTuple], scheduling_plan: List[int]):
        for task, action in zip(tasks, scheduling_plan):
            _, task_attr = task
            if action == ExecutionType.EDGE:
                yield self.sim_env.process(self.execute_remote(task_attr))
            else:
                yield self.sim_env.process(self.execute_local(task_attr))

    def simulate(
            self,
            tasks: List[TaskTuple],
            scheduling_plan: List[int]
    ) -> List[TaskExecution]:
        self.running_simulation_process = self.sim_env.process(self.task_manager(tasks, scheduling_plan))

        try:
            self.sim_env.run(until=self.running_simulation_process)
        except simpy.Interrupt as interrupt:
            print(f"Simulation interrupted: {interrupt.cause}")

        return self.task_info

    @staticmethod
    def build(cluster: Cluster):
        return Simulator(cluster)
