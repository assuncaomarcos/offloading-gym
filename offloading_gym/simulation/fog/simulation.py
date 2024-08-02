#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union
from dataclasses import dataclass

import simpy

from offloading_gym.envs.workload import FogTaskAttr

from .resources import ComputingEnvironment, GeolocationResource
from .config import ResourceType


@dataclass
class TaskRunInfo:
    task_id: int
    resource_id: int
    resource_type: ResourceType
    finish_time: float
    make_span: float
    energy: float


class FogSimulation:
    sim_env: simpy.Environment
    comp_env: ComputingEnvironment
    simulation_process: Union[simpy.Event, None]
    task_info: List[TaskRunInfo]

    def __init__(self, comp_env: ComputingEnvironment):
        self.comp_env = comp_env
        self.sim_env = comp_env.simpy_env
        self.simulation_process = None

    def upload_task(self, num_bytes: int):
        ...

    def download_input_data(self, num_bytes: int):
        ...

    def execute_task(self, task: FogTaskAttr, resource: GeolocationResource):
        with resource.request() as req:
            yield req
            start_time = self.sim_env.now
            task.resource_id = resource.resource_id
            yield self.sim_env.timeout(task.processing_demand / resource.cpu_core_speed)
            finish_time = self.sim_env.now
            energy = self._compute_energy(task, resource)
            self.task_info.append(
                TaskRunInfo(
                    task_id=task.task_id,
                    resource_id=resource.resource_id,
                    finish_time=finish_time,
                    make_span=finish_time - start_time,
                    energy=energy,
                    resource_type=resource.resource_type,
                )
            )

    def _compute_energy(
            self,
            task: FogTaskAttr,
            resource: GeolocationResource
    ) -> float:
        return 0.0

    def task_manager(self, tasks: List[FogTaskAttr], target_resources: List[int]):
        for task_attr, resource_id in zip(tasks, target_resources):
            resource = self.comp_env.comp_resources[resource_id]
            yield self.sim_env.process(self.execute_task(task_attr, resource))

    def simulate(
        self, tasks: List[FogTaskAttr], target_resources: List[int]
    ) -> List[TaskRunInfo]:
        sim_env = self.comp_env.simpy_env
        self.simulation_process = sim_env.process(
            self.task_manager(tasks, target_resources)
        )

        try:
            sim_env.run(until=self.simulation_process)
        except simpy.Interrupt as interrupt:
            print(f"Simulation interrupted: {interrupt.cause}")

        return self.task_info

    @staticmethod
    def build(comp_env: ComputingEnvironment):
        return FogSimulation(comp_env)
