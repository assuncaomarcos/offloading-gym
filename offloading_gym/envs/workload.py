#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..workload import daggen
from ..workload.daggen import DataAttributes
from ..workload.base import Workload
from ..task_graph import TaskGraph, TaskAttr, EdgeAttr
from typing import Optional, List, Dict, Any, Tuple
from networkx import DiGraph
import random

__all__ = ["RandomDAGParameters", "RandomDAGWorkload", "DEFAULT_WORKLOAD_CONFIG"]

DEFAULT_WORKLOAD_CONFIG = {
    "min_datasize": 5120,  # Each task produces between 5KB and 50KB of data
    "max_datasize": 51200,
    "min_computing": 1.0e7,  # Each task requires between 10^7 and 10^8 cycles
    "max_computing": 1.0e8,
    "density_values": [0.2, 0.8],
    "regularity_values": [0.2, 0.8],
    "fat_values": [0.1, 0.4, 0.8],
    "ccr_values": [0.1, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0],
    "jump_values": [1, 2, 4],
}


class RandomDAGParameters:
    def __init__(self, **params) -> None:
        self.min_datasize, self.max_datasize = self._min_max_values(
            params, "min_datasize", "max_datasize"
        )
        self.min_computing, self.max_computing = self._min_max_values(
            params, "min_computing", "max_computing"
        )
        self.density_values = self._list_value(params, "density_values")
        self.regularity_values = self._list_value(params, "regularity_values")
        self.fat_values = self._list_value(params, "fat_values")
        self.ccr_values = self._list_value(params, "ccr_values")
        self.jump_values = self._list_value(params, "jump_values")

    def _min_max_values(
        self, parameters: Dict, min_param_name: str, max_param_name: str
    ) -> Tuple[Any, Any]:
        min_value = self._get_value(parameters, min_param_name)
        max_value = self._get_value(parameters, max_param_name)
        assert (
            0 <= min_value < max_value
        ), f"Unsupported argument {min_value} or {max_value}"
        return min_value, max_value

    @staticmethod
    def _get_value(parameters: Dict, param_name: str) -> Any:
        return parameters.get(param_name, DEFAULT_WORKLOAD_CONFIG[param_name])

    def _list_value(self, parameters: Dict, param_name: str) -> Any:
        param_value = self._get_value(parameters, param_name)
        if not isinstance(param_value, list):
            raise ValueError(f"Parameter {param_name} must be a list")

        return param_value


DEFAULT_WORKLOAD = RandomDAGParameters(**DEFAULT_WORKLOAD_CONFIG)


class RandomDAGWorkload(Workload):
    dag_parameters: RandomDAGParameters
    tasks_per_app: int
    random_instance: random.Random

    def __init__(
        self,
        tasks_per_app: int,
        random_seed: int = None,
        dag_parameters: RandomDAGParameters = DEFAULT_WORKLOAD,
        length: int = 0,
    ):
        super().__init__(length)
        self.dag_parameters = dag_parameters
        self.tasks_per_app = tasks_per_app
        self.random_instance = (
            random.Random(random_seed) if random_seed is not None else random
        )

    def _random_dag_parameters(self):
        return {
            "density": self.random_instance.choice(self.dag_parameters.density_values),
            "fat": self.random_instance.choice(self.dag_parameters.fat_values),
            "regularity": self.random_instance.choice(
                self.dag_parameters.regularity_values
            ),
            "ccr": self.random_instance.choice(self.dag_parameters.ccr_values),
            "jump": self.random_instance.choice(self.dag_parameters.jump_values),
        }

    def step(self, offset=1) -> List[Optional[TaskGraph]]:
        self.current_time += offset
        dag_params = self._random_dag_parameters()
        dag = daggen.random_dag(num_tasks=self.tasks_per_app, **dag_params)
        return [self._create_task_graph(dag)]

    def _create_task_graph(self, graph: DiGraph) -> TaskGraph:
        datasize = self.random_instance.uniform(
            self.dag_parameters.min_datasize, self.dag_parameters.max_datasize
        )
        computing = self.random_instance.uniform(
            self.dag_parameters.min_computing, self.dag_parameters.max_computing
        )
        tasks, edges = [], []

        for node_id, data in graph.nodes.items():
            task_comp = int(data[DataAttributes.PROCESSING_COST] * computing)
            tasks.append(
                (node_id, TaskAttr(task_id=node_id, processing_demand=task_comp))
            )
        for sd, data in graph.edges.items():
            src, dst = sd
            edges.append(
                (
                    src,
                    dst,
                    EdgeAttr(
                        datasize=int(data[DataAttributes.COMMUNICATION_COST] * datasize)
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
