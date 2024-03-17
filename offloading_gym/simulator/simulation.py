#!/usr/bin/env python
# -*- coding: utf-8 -*-

import simpy
from .cluster import Cluster
from ..task_graph import TaskAttr, TaskTuple
from typing import List, Dict
from enum import IntEnum
from dataclasses import dataclass
from collections import OrderedDict


class ExecutionType(IntEnum):
    LOCAL = 0
    EDGE = 1


@dataclass
class TaskExecution:
    task_id: int
    finish_time: float


class Simulator:
    cluster: Cluster
    sim_env: simpy.Environment
    local_device: simpy.Resource
    edge_server: simpy.Resource
    task_info: OrderedDict[int, TaskExecution]

    def __init__(self, cluster: Cluster):
        self.sim_env = simpy.Environment()
        self.cluster = cluster

        # Assuming one task at a time for now to be compliant with the MRLCO paper
        self.edge_server = simpy.Resource(self.sim_env, capacity=1)
        self.local_device = simpy.Resource(self.sim_env, capacity=1)
        self.task_info = OrderedDict()

    def upload(self, data_size):
        yield self.sim_env.timeout(self.cluster.upload_time(data_size))

    def download(self, data_size):
        yield self.sim_env.timeout(self.cluster.download_time(data_size))

    def execute_local(self, task: TaskAttr):
        with self.local_device.request() as req:
            yield req
            yield self.sim_env.timeout(self.cluster.local_execution_time(task.processing_demand))
            self.task_info[task.task_id] = TaskExecution(task_id=task.task_id, finish_time=self.sim_env.now)

    def execute_remote(self, task: TaskAttr):
        with self.edge_server.request() as req:
            yield req
            yield self.sim_env.process(self.upload(task.task_size))
            yield self.sim_env.timeout(self.cluster.edge_execution_time(task.processing_demand))
            yield self.sim_env.process(self.download(task.output_datasize))
            self.task_info[task.task_id] = TaskExecution(task_id=task.task_id, finish_time=self.sim_env.now)

    def task_manager(self, tasks: List[TaskTuple], scheduling_plan: List[int]):
        for task, action in zip(tasks, scheduling_plan):
            _, task_attr = task
            if action == ExecutionType.EDGE:
                yield self.sim_env.process(self.execute_remote(task_attr))
            else:
                yield self.sim_env.process(self.execute_local(task_attr))

    def simulate(self, tasks: List[TaskTuple], scheduling_plan: List[int]) -> Dict[int, TaskExecution]:
        self.sim_env.process(self.task_manager(tasks, scheduling_plan))
        self.sim_env.run()
        return self.task_info

    @staticmethod
    def build(cluster: Cluster):
        return Simulator(cluster)
