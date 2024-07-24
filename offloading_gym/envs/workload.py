#!/usr/bin/env python
# -*- coding: utf-8 -*-

from offloading_gym.workload import RandomDAGGenerator
from offloading_gym.simulation.fog import config as fog_config
from offloading_gym.task_graph import TaskGraph, TaskAttr

__all__ = ["RandomGraphWorkload", "FogDAGWorkload"]


class RandomGraphWorkload(RandomDAGGenerator):

    def __init__(self, length: int = 0, **kwargs):
        super().__init__(length, **kwargs)

    @classmethod
    def build(cls, args: dict):
        kwargs = {k: v for k, v in args.items()}
        return RandomGraphWorkload(**kwargs)


class FogDAGWorkload(RandomDAGGenerator):
    min_memory: float
    max_memory: float

    def __init__(self, length: int = 0, **kwargs):
        super().__init__(length, **kwargs)
        self.min_memory = kwargs.get("min_memory", 0.0)
        self.max_memory = kwargs.get("max_memory", 0.0)
        assert 0.0 < self.min_memory <= self.max_memory, "min_memory must > 0.0 and <= max_memory"
        assert 0.0 < self.max_memory, "max_memory must be > 0.0"

    def random_task_graph(self) -> TaskGraph:
        dag = super().random_task_graph()
        rng = self.np_random
        for node_id, data in dag.nodes.items():
            data["memory"] = rng.uniform(self.min_memory, self.max_memory)
            
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




