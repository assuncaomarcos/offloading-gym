#!/usr/bin/env python
# -*- coding: utf-8 -*-

from offloading_gym.workload import RandomDAGGenerator
from offloading_gym.simulation.fog import config as fog_config
from offloading_gym.task_graph import TaskGraph, TaskAttr

__all__ = ["RandomGraphWorkload", "FogDAGWorkload", "FogTaskAttr"]


class RandomGraphWorkload(RandomDAGGenerator):

    def __init__(self, length: int = 0, **kwargs):
        super().__init__(length, **kwargs)

    @classmethod
    def build(cls, args: dict):
        kwargs = {k: v for k, v in args.items()}
        return RandomGraphWorkload(**kwargs)


class FogTaskAttr(TaskAttr):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['resource_id'] = -1

    @property
    def memory(self) -> int:
        """The amount of memory in bytes the task requires."""
        return self['memory']

    @memory.setter
    def memory(self, value: int) -> None:
        """Set the amount of memory in bytes."""
        self['memory'] = value

    @property
    def rank(self) -> float:
        """The task rank."""
        return self['rank']

    @rank.setter
    def rank(self, value: float) -> None:
        """Set the task rank."""
        self['rank'] = value

    @property
    def resource_id(self) -> int:
        """
        The id of the resource to which the task has been assigned ;
        -1 if unknown.
        """
        return self['resource_id']

    @resource_id.setter
    def resource_id(self, value: int) -> None:
        """Sets the id of the resource to which the task has been assigned"""
        self['resource_id'] = value


class FogDAGWorkload(RandomDAGGenerator):
    min_memory: int
    max_memory: int
    task_attr_factory = FogTaskAttr

    def __init__(self, length: int = 0, **kwargs):
        super().__init__(length, **kwargs)
        self.min_memory = kwargs.get("min_memory", 0)
        self.max_memory = kwargs.get("max_memory", 0)
        assert 0 < self.min_memory <= self.max_memory, "min_memory must > 0.0 and <= max_memory"
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




