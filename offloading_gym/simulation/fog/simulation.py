#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union
from dataclasses import dataclass

import simpy

from offloading_gym.envs.workload import FogTaskAttr

from .resources import ComputingEnvironment, GeolocationResource
from .config import ResourceType

IOT_DEVICE_ID = 0
GB_IN_BYTES = 1024 ** 3


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
    iot_device: GeolocationResource

    def __init__(self, comp_env: ComputingEnvironment):
        self.comp_env = comp_env
        self.sim_env = comp_env.simpy_env
        self.simulation_process = None
        self.iot_device = comp_env.compute_resources.get(IOT_DEVICE_ID)
        self.task_info = []

    def _execute_task(self, task: FogTaskAttr, resource: GeolocationResource):
        with resource.request(cpu_cores=1, memory=task.memory / GB_IN_BYTES) as req:
            yield req
            start_time = self.sim_env.now
            task.resource_id = resource.resource_id

            upload_delay = self.comp_env.data_transfer_time(
                source=self.iot_device,
                destination=resource,
                num_bytes=task.task_size,
            )
            download_delay = self.comp_env.data_transfer_time(
                source=resource,
                destination=self.iot_device,
                num_bytes=task.output_datasize,
            )
            task_runtime = self.comp_env.task_runtime(resource, task)

            yield self.sim_env.timeout(upload_delay)
            yield self.sim_env.timeout(task_runtime)
            yield self.sim_env.timeout(download_delay)

            finish_time = self.sim_env.now
            energy = self.comp_env.energy_use(
                resource=resource,
                task_runtime=task_runtime,
                task_comm_time=upload_delay + download_delay
            )

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

    def _task_manager(self, tasks: List[FogTaskAttr], target_resources: List[int]):
        for task_attr, resource_id in zip(tasks, target_resources):
            resource = self.comp_env.compute_resources[resource_id]
            yield self.sim_env.process(self._execute_task(task_attr, resource))

    def simulate(
        self, tasks: List[FogTaskAttr], target_resources: List[int]
    ) -> List[TaskRunInfo]:
        sim_env = self.comp_env.simpy_env
        self.simulation_process = sim_env.process(
            self._task_manager(tasks, target_resources)
        )

        try:
            sim_env.run(until=self.simulation_process)
        except simpy.Interrupt as interrupt:
            print(f"Simulation interrupted: {interrupt.cause}")

        return self.task_info

    @staticmethod
    def build(comp_env: ComputingEnvironment):
        return FogSimulation(comp_env)
