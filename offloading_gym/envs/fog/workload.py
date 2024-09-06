#!/usr/bin/env python
# -*- coding: utf-8 -*-

from offloading_gym.workload import RandomDAGGenerator
from offloading_gym.simulation.fog import typing as fog_config
from offloading_gym.task_graph import TaskGraph, TaskAttr

__all__ = ["FogDAGWorkload", "FogTaskAttr"]


class FogTaskAttr(TaskAttr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["resource_id"] = -1
        self["arrival_time"] = 0.0
        self["start_time"] = 0.0
        self["finish_time"] = 0.0
        self["upload_delay"] = 0.0
        self["download_delay"] = 0.0
        self["energy_used"] = 0.0

    @property
    def memory(self) -> int:
        """The amount of memory in bytes the task requires."""
        return self["memory"]

    @memory.setter
    def memory(self, value: int) -> None:
        """Set the amount of memory in bytes."""
        self["memory"] = value

    @property
    def rank(self) -> float:
        """The task rank."""
        return self["rank"]

    @rank.setter
    def rank(self, value: float) -> None:
        """Set the task rank."""
        self["rank"] = value

    @property
    def resource_id(self) -> int:
        """
        The id of the resource to which the task has been assigned ;
        -1 if unknown.
        """
        return self["resource_id"]

    @resource_id.setter
    def resource_id(self, value: int) -> None:
        """Sets the id of the resource to which the task has been assigned"""
        self["resource_id"] = value

    @property
    def arrival_time(self) -> float:
        """The arrival time of the task."""
        return self["arrival_time"]

    @arrival_time.setter
    def arrival_time(self, value: float) -> None:
        """Sets the arrival time of the task."""
        self["arrival_time"] = value

    @property
    def start_time(self) -> float:
        """The start time of the task."""
        return self["start_time"]

    @start_time.setter
    def start_time(self, value: float) -> None:
        """Sets the start time of the task."""
        self["start_time"] = value

    @property
    def upload_delay(self) -> float:
        """The upload delay of the task."""
        return self["upload_delay"]

    @upload_delay.setter
    def upload_delay(self, value: float) -> None:
        """Sets the upload delay of the task."""
        self["upload_delay"] = value

    @property
    def download_delay(self) -> float:
        """The download delay of the task."""
        return self["download_delay"]

    @download_delay.setter
    def download_delay(self, value: float) -> None:
        """Sets the download delay of the task."""
        self["download_delay"] = value

    @property
    def finish_time(self) -> float:
        """The finish time of the task."""
        return self["finish_time"]

    @finish_time.setter
    def finish_time(self, value: float) -> None:
        """Sets the finish time of the task."""
        self["finish_time"] = value

    @property
    def makespan(self) -> float:
        """The makespan of the task."""
        return self.finish_time - self.arrival_time

    @property
    def runtime(self) -> float:
        """The runtime of the task."""
        return (
            self.finish_time - self.start_time - self.upload_delay - self.download_delay
        )

    @property
    def communication_time(self) -> float:
        """The communication time of the task."""
        return self.upload_delay + self.download_delay

    @property
    def energy_used(self) -> float:
        """The energy used by the task."""
        return self["energy_used"]

    @energy_used.setter
    def energy_used(self, value: float) -> None:
        """Sets the energy used by the task."""
        self["energy_used"] = value


class FogDAGWorkload(RandomDAGGenerator):
    min_memory: int
    max_memory: int
    task_attr_factory = FogTaskAttr

    def __init__(self, length: int = 0, **kwargs):
        super().__init__(length, **kwargs)
        self.min_memory = kwargs.get("min_memory", 0)
        self.max_memory = kwargs.get("max_memory", 0)
        assert (
            0 < self.min_memory <= self.max_memory
        ), "min_memory must > 0.0 and <= max_memory"
        assert 0 < self.max_memory, "max_memory must be > 0"

    def random_task_graph(self) -> TaskGraph:
        dag = super().random_task_graph()
        rng = self.np_random
        for node_id, data in dag.nodes.items():
            data.memory = rng.integers(self.min_memory, self.max_memory, endpoint=True)

        return dag

    @staticmethod
    def build(args):
        if isinstance(args, dict):
            return FogDAGWorkload._build_from_dict(args)
        elif isinstance(args, fog_config.WorkloadConfig):
            return FogDAGWorkload._build_from_config(args)
        else:
            raise NotImplementedError("Unsupported configuration object")

    @staticmethod
    def _build_from_dict(args: dict):
        kwargs = {k: v for k, v in args.items()}
        return FogDAGWorkload(**kwargs)

    @staticmethod
    def _build_from_config(args: fog_config.WorkloadConfig):
        return FogDAGWorkload._build_from_dict(args.as_dict())
