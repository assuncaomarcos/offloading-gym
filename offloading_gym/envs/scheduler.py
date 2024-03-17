#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

from numpy.typing import NDArray
from enum import IntEnum
import numpy as np

from ..simulator import Scheduler
from ..task_graph import TaskAttr

DEFAULT_SCHEDULER_CONFIG = {
    "type": "fifo_scheduler",
    "num_edge_cpus": 10,
    "edge_cpu_capacity": 1024 * 1024,
    "user_cpu_capacity": 1024 * 1024,
    "upload_rate": 10,
    "download_rate": 10
}


class ExecutionType(IntEnum):
    LOCAL = 0
    EDGE = 1


class TaskSchedule:
    task: TaskAttr
    start_time: float
    finish_time: float

    def __init__(self, task: TaskAttr):
        self.task = task
        self.start_time = 0
        self.running_time = 0

    @property
    def finish_time(self):
        return self.start_time + self.running_time

    def reset(self):
        self.start_time = 0
        self.running_time = 0


class DefaultScheduler(Scheduler):
    _energy_use: float
    edge_avail_time: float
    link_avail_time: float
    local_avail_time: float
    tasks: List[TaskSchedule]

    def add_tasks(self, tasks: List[TaskAttr]):
        for task in tasks:
            self.tasks.append(TaskSchedule(task))

    def compute_schedule(self, tasks: List[TaskAttr], scheduling_plan: NDArray[np.int8]):
        self.cluster.reset()
        self._reset()
        for action, task_attr in zip(scheduling_plan, tasks):
            if action == ExecutionType.LOCAL:
                pass
            else:
                pass

    def local_execution(self, task_attr: TaskAttr):
        runtime = self.cluster.local_execution_time(task_attr.processing_demand)

    def reset(self):
        pass

    # @staticmethod
    # def compute_times(
    #         task_idx, task_graph, task, available_time,
    #         execution_cost_func, ft_resource, ft_channel
    # ):
    #     if len(task_graph.predecessors[task_idx]) != 0:
    #         start_time = max(available_time, max(
    #             [max(ft_resource[j], ft_channel[j]) for j in task_graph.predecessors[task_idx]]
    #         ))
    #     else:
    #         start_time = available_time
    #
    #     running_time = execution_cost_func(task.proc_datasize)
    #     finish_time = start_time + running_time
    #     return [start_time, running_time, finish_time]


    def _reset(self):
        self.edge_avail_time = 0.0
        self.link_avail_time = 0.0
        self.local_avail_time = 0.0

    @staticmethod
    def build(**kwargs):
        return DefaultScheduler(**kwargs)

    # def get_scheduling_cost_step_by_step(self, plan, task_graph):
    #     cloud_available_time = 0.0
    #     ws_available_time = 0.0
    #     local_available_time = 0.0
    #     task_number = task_graph.number_of_tasks
    #
    #     time_local = [0] * task_number  # run time on local processor
    #     time_ul = [0] * task_number  # run time on sending channel
    #     time_dl = [0] * task_number  # run time on receiving channel
    #
    #     ft_cloud = [0] * task_number  # finish time on cloud for each task
    #     ft_ws = [0] * task_number  # finish time on sending channel for each task
    #     ft_local = [0] * task_number  # local finish time for each task
    #     ft_wr = [0] * task_number  # finish time in the receiving channel for each task
    #
    #     current_ft = total_energy = 0.0
    #     return_latency, return_energy = [], []
    #
    #     for item in plan:
    #         task_idx = item[0]
    #         task = task_graph.tasks[task_idx]
    #         action = item[1]
    #
    #         # Local execution
    #         if action == 0:
    #             # Compute the local finish time
    #             _, time_local[task_idx], ft_local[task_idx] = self.compute_times(
    #                 task_idx, task_graph, task, local_available_time,
    #                 self.resource_cluster.locally_execution_cost, ft_local, ft_wr
    #             )
    #
    #             local_available_time = ft_local[task_idx]
    #             task_finish_time = ft_local[task_idx]
    #             energy_consumption = self.compute_local_energy(time_local[task_idx])
    #
    #         # Offloading
    #         else:
    #
    #             # Compute the remote finish time
    #             _, time_ul[task_idx], ft_ws[task_idx] = self.compute_times(
    #                 task_idx, task_graph, task, ws_available_time,
    #                 self.resource_cluster.up_transmission_cost, ft_local, ft_ws
    #             )
    #             ws_available_time = ft_ws[task_idx]
    #
    #             cloud_start_time = cloud_available_time
    #             if not len(task_graph.predecessors[task_idx]):
    #                 cloud_start_time = max(cloud_available_time, ft_ws[task_idx])
    #
    #             # TODO: Not sure whether it's the processing or transmission data size
    #             _, _, ft_cloud[task_idx] = self.compute_times(
    #                 task_idx, task_graph, task, cloud_start_time,
    #                 self.resource_cluster.mec_execution_cost, ft_cloud, ft_ws
    #             )
    #             cloud_available_time = ft_cloud[task_idx]
    #             wr_start_time = cloud_available_time
    #
    #             time_dl[task_idx] = self.resource_cluster.dl_transmission_cost(task.trans_datasize)
    #             ft_wr[task_idx] = wr_start_time + time_dl[task_idx]
    #             task_finish_time = ft_wr[task_idx]
    #             energy_consumption = self.compute_offloading_energy(time_ul[task_idx], self.ptx + time_dl[task_idx])
    #
    #         current_ft = max(task_finish_time, current_ft)
    #         total_energy += energy_consumption
    #         delta_make_span = max(task_finish_time, current_ft) - current_ft
    #         delta_energy = energy_consumption
    #
    #         return_latency.append(delta_make_span)
    #         return_energy.append(delta_energy)
    #
    #     return return_latency, return_energy, current_ft, total_energy


def build_scheduler(scheduler_config: dict):
    scheduler_type = scheduler_config['type']
    kwargs = {k: v for k, v in scheduler_config.items() if k != 'type'}
    if scheduler_type == 'fifo_scheduler':
        return DefaultScheduler.build(**kwargs)
    else:
        raise RuntimeError(f'Unsupported scheduler type {scheduler_type}')


