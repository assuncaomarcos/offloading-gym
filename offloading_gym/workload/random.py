#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Dict, Any, Tuple
from ..task_graph import TaskGraph, TaskAttr, EdgeAttr
from .daggen import random_dag
from .base import Workload
import random


class RandomDAGGenerator(Workload):

    def __init__(self, length: int = 0, **kwargs):
        super().__init__(length=length)
        self.num_tasks = self.get_value(kwargs, "num_tasks")
        self.min_datasize, self.max_datasize = self.min_max(
            kwargs, "min_datasize", "max_datasize"
        )
        self.min_computing, self.max_computing = self.min_max(
            kwargs, "min_computing", "max_computing"
        )
        self.densities = self.list(kwargs, "density_values")
        self.regularities = self.list(kwargs, "regularity_values")
        self.fat_values = self.list(kwargs, "fat_values")
        self.ccr_values = self.list(kwargs, "ccr_values")
        self.jumps = self.list(kwargs, "jump_values")

    def min_max(
        self, parameters: Dict, min_param_name: str, max_param_name: str
    ) -> Tuple[Any, Any]:
        min_value = self.get_value(parameters, min_param_name)
        max_value = self.get_value(parameters, max_param_name)
        assert (
            0 <= min_value < max_value
        ), f"Unsupported argument {min_value} or {max_value}"
        return min_value, max_value

    @staticmethod
    def get_value(parameters: Dict, param_name: str) -> Any:
        return parameters.get(param_name, None)

    def list(self, parameters: Dict, param_name: str) -> Any:
        param_value = self.get_value(parameters, param_name)
        if not isinstance(param_value, list):
            raise ValueError(f"Parameter {param_name} must be a list")

        return param_value

    def random_daggen_params(self):
        return {
            "rng": self.np_random,
            "num_tasks": self.num_tasks,
            "density": random.choice(self.densities),
            "fat": random.choice(self.fat_values),
            "regularity": random.choice(self.regularities),
            "ccr": random.choice(self.ccr_values),
            "jump": random.choice(self.jumps),
        }

    def step(self, *, offset: Optional[int] = 1) -> List[Optional[TaskGraph]]:
        self.current_time += offset
        return [self.random_task_graph()]

    def random_task_graph(self) -> TaskGraph:
        dag_params = self.random_daggen_params()
        dag = random_dag(**dag_params)
        tasks, edges = [], []

        for node_id, data in dag.nodes.items():
            tasks.append(
                (
                    node_id,
                    TaskAttr(
                        task_id=node_id,
                        processing_demand=data["processing_demand"],
                        task_size=data["output_datasize"],
                        output_datasize=data["output_datasize"],
                    ),
                )
            )
        for sd, data in dag.edges.items():
            src, dst = sd
            edges.append(
                (
                    src,
                    dst,
                    EdgeAttr(datasize=data["datasize"]),
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
