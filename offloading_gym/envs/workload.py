#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..workload import daggen
from ..workload.daggen import TaskDataAttr
from ..workload.base import Workload
from ..task_graph import TaskGraph, TaskAttr, EdgeAttr
from typing import Optional, List, Dict, Any, Tuple
from random import Random

__all__ = [
    "RandomDAGWorkload",
    "DEFAULT_WORKLOAD_CONFIG"
]

DEFAULT_WORKLOAD_CONFIG = {
    "type": "random_dag",
    "num_tasks": 20,
    "min_datasize": 5120,    # Each task produces between 5KB and 50KB of data
    "max_datasize": 51200,
    "min_computing": 1.0e7,  # Each task requires between 10^7 and 10^8 cycles
    "max_computing": 1.0e8,
    "density_values": [0.2, 0.8],
    "regularity_values": [0.2, 0.8],
    "fat_values": [0.1, 0.4, 0.8],
    "ccr_values": [0.1, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0],
    "jump_values": [1, 2, 4],
}


class RandomDAGWorkload(Workload):
    random: Random

    def __init__(self, random_seed: int = 0, length: int = 0, **kwargs):
        super().__init__(length)
        self.random = Random(random_seed)
        self.num_tasks = self.get_value(kwargs, "num_tasks")
        self.min_datasize, self.max_datasize = self.min_max(kwargs, "min_datasize", "max_datasize")
        self.min_computing, self.max_computing = self.min_max(kwargs, "min_computing", "max_computing")
        self.densities = self.list(kwargs, "density_values")
        self.regularities = self.list(kwargs, "regularity_values")
        self.fat_values = self.list(kwargs, "fat_values")
        self.ccr_values = self.list(kwargs, "ccr_values")
        self.jumps = self.list(kwargs, "jump_values")

    def min_max(self, parameters: Dict, min_param_name: str, max_param_name: str) -> Tuple[Any, Any]:
        min_value = self.get_value(parameters, min_param_name)
        max_value = self.get_value(parameters, max_param_name)
        assert 0 <= min_value < max_value, f"Unsupported argument {min_value} or {max_value}"
        return min_value, max_value

    @staticmethod
    def get_value(parameters: Dict, param_name: str) -> Any:
        return parameters.get(param_name, DEFAULT_WORKLOAD_CONFIG[param_name])

    def list(self, parameters: Dict, param_name: str) -> Any:
        param_value = self.get_value(parameters, param_name)
        if not isinstance(param_value, list):
            raise ValueError(f"Parameter {param_name} must be a list")

        return param_value

    def random_daggen_params(self):
        return {
            "num_tasks": self.num_tasks,
            "density": self.random.choice(self.densities),
            "fat": self.random.choice(self.fat_values),
            "regularity": self.random.choice(self.regularities),
            "ccr": self.random.choice(self.ccr_values),
            "jump": self.random.choice(self.jumps),
        }

    @staticmethod
    def build(**kwargs):
        return RandomDAGWorkload(**kwargs)

    def seed(self, seed: int = None):
        self.random.seed(seed)

    def step(self, offset=1) -> List[Optional[TaskGraph]]:
        self.current_time += offset
        return [self.generate_task_graph()]

    def generate_task_graph(self) -> TaskGraph:
        dag_params = self.random_daggen_params()
        dag = daggen.random_dag(**dag_params)

        datasize = self.random.uniform(self.min_datasize, self.max_datasize)
        computing = self.random.uniform(self.min_computing, self.max_computing)
        tasks, edges = [], []

        for node_id, data in dag.nodes.items():
            task_comp = int(data[TaskDataAttr.PROCESSING_COST] * computing)
            tasks.append(
                (node_id, TaskAttr(task_id=node_id, processing_demand=task_comp))
            )
        for sd, data in dag.edges.items():
            src, dst = sd
            edges.append(
                (
                    src,
                    dst,
                    EdgeAttr(
                        datasize=int(data[TaskDataAttr.COMMUNICATION_COST] * datasize)
                    ),
                )
            )

        task_graph = TaskGraph()
        task_graph.add_nodes_from(tasks)
        task_graph.add_edges_from(edges)
        return task_graph

    def __len__(self):
        return self.length

    def peek(self):
        return self.step(0)


def build_workload(workload_config: dict):
    wkl_type = workload_config['type']
    kwargs = {k: v for k, v in workload_config.items() if k != 'type'}
    if wkl_type == 'random_dag':
        return RandomDAGWorkload.build(**kwargs)
    else:
        raise RuntimeError(f'Unsupported workload type {wkl_type}')
