#!/usr/bin/env python
# -*- coding: utf-8 -*-

import simpy
from typing import List, Union
from dataclasses import dataclass
from resources import ComputingEnvironment, GeolocationResource
from offloading_gym.task_graph import TaskAttr, TaskTuple
from .config import ResourceType


@dataclass
class TaskRunInfo:
    task_id: int
    resource_id: int
    resource_type: ResourceType
    finish_time: float
    make_span: float
    energy: float


class Simulation:
    sim_env: simpy.Environment
    comp_env: ComputingEnvironment
    simulation_process: Union[simpy.Event, None]
    task_info: List[TaskRunInfo]

    def __init__(self, comp_env: ComputingEnvironment):
        self.comp_env = comp_env
        self.sim_env = comp_env.simpy_env
        self.simulation_process = None

    def execute_task(self, task: TaskAttr, resource: GeolocationResource):
        with resource.request() as req:
            yield req
            start_time = self.sim_env.now
            yield self.sim_env.timeout(task.processing_demand / resource.cpu_core_speed)
            finish_time = self.sim_env.now

    def task_manager(self, tasks: List[TaskTuple], scheduling: List[int]):
        for task, resource_id in zip(tasks, scheduling):
            _, task_attr = task

    def simulate(
        self, tasks: List[TaskTuple], scheduling_plan: List[int]
    ) -> List[TaskRunInfo]:
        sim_env = self.comp_env.simpy_env
        self.simulation_process = sim_env.process(
            self.task_manager(tasks, scheduling_plan)
        )

        try:
            sim_env.run(until=self.simulation_process)
        except simpy.Interrupt as interrupt:
            print(f"Simulation interrupted: {interrupt.cause}")

        return self.task_info
