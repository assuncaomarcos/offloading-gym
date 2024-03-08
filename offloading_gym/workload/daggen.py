#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module generates Directed Acyclic Graphs (DAGs) based on parameters like task number,
networking features, and Communication to Computation Ratio (CCR). This is useful in
simulating task scheduling in parallel and distributed systems.

This module is based on daggen random `graph generator <https://github.com/frs69wq/daggen>` _
proposed by Suter & Hunold and the changes proposed by Arabnejad and Barbosa.
For more details, see the article below:

H. Arabnejad and J. Barbosa, List Scheduling Algorithm for Heterogeneous Systems by
an Optimistic Cost Table, IEEE Transactions on Parallel and Distributed Systems,
Vol. 25, N. 3, March 2014.

By default, this module uses Python's singleton pseudo-random number generator to generate
random numbers.

Example:
    >>> from random import random
    >>> import networkx as nx
    >>> from offloading_gym.workload import daggen
    >>>
    >>> graph = daggen.random_dag(num_tasks=20, ccr=0.5)
    >>> print(graph)
    DiGraph with 20 nodes and 25 edges
"""

import math
from networkx import DiGraph
from dataclasses import dataclass
from typing import List, Any, Tuple
from enum import Enum
import random

__all__ = [
    "random_dag",
    "TaskDataAttr"
]


class TaskDataAttr(Enum):
    TASK_ID = "task_id"
    PROCESSING_COST = "processing_cost"
    COMMUNICATION_COST = "communication_cost"

    def __str__(self):
        return self.value


# Default values for generated DAGs
NUM_TASKS = 20
FAT = 0.7
REGULARITY = 0.5
DENSITY = 0.6
JUMP_SIZE = 1
CCR = 0.3

COST_PRECISION = 4


@dataclass
class TaskInfo:
    task_id: int
    computing_cost: float
    data_cost: float
    n_children: int
    children: List[Any]


def random_int_in_range(num, range_percent: float) -> int:
    """Generates a random integer within a range around a specified number."""
    r = -range_percent + (2 * range_percent * random.random())
    new_int = max(1, int(num * (1.0 + r / 100.00)))
    return new_int


def create_tasks(
    n_tasks: int, fat: float, regularity: float
) -> Tuple[List[List[TaskInfo]], float]:
    # Compute the number of tasks per level
    n_tasks_per_level = int(fat * math.sqrt(n_tasks))
    total_tasks = 0
    total_comp_cost = 0.0
    tasks = []
    sampled_cost = random.uniform(0.0, 1.0)

    while total_tasks < n_tasks:
        n_tasks_at_level = min(
            random_int_in_range(n_tasks_per_level, 100.0 - 100.0 * regularity),
            n_tasks - total_tasks,
        )
        comp_cost = round(random.uniform(0.0, 2 * sampled_cost), COST_PRECISION)
        total_comp_cost += comp_cost
        tasks_at_level = [
            TaskInfo(task_id, comp_cost, 0, 0, [])
            for task_id in range(total_tasks + 1, total_tasks + 1 + n_tasks_at_level)
        ]
        tasks.append(tasks_at_level)
        total_tasks += n_tasks_at_level

    return tasks, total_comp_cost


def create_dependencies(tasks: List[List[TaskInfo]], density: float, jump: int) -> None:
    n_levels = len(tasks)

    # For all levels but the last one
    for level in range(1, n_levels):
        for level_idx in range(len(tasks[level])):
            # Compute how many parents the task should have
            n_tasks_upper_level = len(tasks[level - 1])
            n_parents = min(
                1 + int(random.uniform(0.0, density * n_tasks_upper_level)),
                n_tasks_upper_level,
            )

            for _ in range(n_parents):
                # compute the level of the parent
                parent_level = max(0, level - int(random.uniform(1.0, jump + 1)))
                parent_idx = int(random.uniform(0, len(tasks[parent_level])))
                parent = tasks[parent_level][parent_idx]

                n_tasks_at_level = child_idx = 0
                # Increment the parent_idx until a slot is found
                for n_tasks_at_level in range(len(tasks[parent_level])):
                    for child_idx in range(parent.n_children):
                        if parent.children[child_idx] == tasks[level][level_idx]:
                            parent_idx = (parent_idx + 1) % len(tasks[parent_level])
                            parent = tasks[parent_level][parent_idx]
                            break
                    if child_idx == parent.n_children - 1:
                        break

                if n_tasks_at_level < len(tasks[parent_level]):
                    # Update the parent's children list
                    parent.children.append(tasks[level][level_idx])
                    parent.n_children += 1


def set_communication_costs(
    tasks: List[List[TaskInfo]], total_comp_cost: float, ccr: float
) -> None:
    total_comm_cost = total_comp_cost * ccr
    tasks_with_children = [
        task for level in tasks for task in level if task.n_children > 0
    ]
    comm_costs = [random.random() * task.n_children for task in tasks_with_children]

    # Adjust costs to match the total communication cost
    sum_edge_costs = sum(comm_costs)
    for task, cost in zip(tasks_with_children, comm_costs):
        task.data_cost = round(
            (cost * total_comm_cost / sum_edge_costs) / task.n_children, COST_PRECISION
        )


def create_task_graph(task_info: List[List[TaskInfo]]) -> DiGraph:
    tasks, edges = [], []
    for sublist in task_info:
        for task in sublist:
            tasks.append(
                (
                    task.task_id,
                    {
                        TaskDataAttr.TASK_ID: task.task_id,
                        TaskDataAttr.PROCESSING_COST: task.computing_cost,
                    },
                )
            )
            for dest in task.children:
                edges.append(
                    (
                        task.task_id,
                        dest.task_id,
                        {TaskDataAttr.COMMUNICATION_COST: task.data_cost},
                    )
                )

    task_graph = DiGraph()
    task_graph.add_nodes_from(tasks)
    task_graph.add_edges_from(edges)
    return task_graph


def random_dag(
        num_tasks: int = NUM_TASKS,
        fat: float = FAT,
        density: float = DENSITY,
        regularity: float = REGULARITY,
        jump: int = JUMP_SIZE,
        ccr: float = CCR,
) -> DiGraph:
    """
    Generates a directed acyclic graph (DAG) representing a task graph.

    Args:
        num_tasks (int): The number of tasks in the task graph. Default is 20.
        fat (float): The fatness factor for task durations. Default is 0.7.
        density (float): The density factor for task dependencies. Default is 0.6.
        regularity (float): The regularity factor for task durations. Default is 0.5.
        jump (int): The jump size for task dependencies. Default is 1.
        ccr (float): The communication-to-computation ratio. Default is 0.3.

    Returns:
        DiGraph: The generated task graph as a NetworkX DiGraph object.

    Raises:
        AssertionError: If any of the input parameters fail the specified constraints.

    """
    assert num_tasks >= 1, "num_tasks must be >= 1"
    assert 0 <= fat, "fat must be greater than or equal to 0"
    assert 0 <= density <= 1, "density must be between 0 and 1"
    assert 0 <= regularity <= 1, "regularity must be between 0 and 1"
    assert 0 <= ccr <= 10, "ccr must be between 0 and 10"
    assert 1 <= jump <= 4, "jump_size must be between 1 and 4"

    tasks, total_comp_cost = create_tasks(
        n_tasks=num_tasks, fat=fat, regularity=regularity
    )
    create_dependencies(tasks=tasks, density=density, jump=jump)
    set_communication_costs(tasks, total_comp_cost, ccr)
    task_graph = create_task_graph(tasks)
    task_graph.graph[TaskDataAttr.PROCESSING_COST] = total_comp_cost
    task_graph.graph[TaskDataAttr.COMMUNICATION_COST] = total_comp_cost * ccr
    return task_graph
