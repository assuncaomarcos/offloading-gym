#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module generates directed acyclic graphs (DAGs) based on parameters like task number,
networking features, and communication to computation ratio (CCR). This is useful in
simulating task scheduling in parallel and distributed systems.

This module uses Python's pseudo-random number generator to generate random numbers.
To ensure reproducibility of results, you should seed the random number generator prior to
invoking this module.
"""

import random
import math
from networkx import DiGraph
from dataclasses import dataclass
from typing import List, Any, Tuple

__all__ = [
    'daggen'
]


# Default values for daggen DAGs
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
    communication_cost: float
    n_children: int
    children: List[Any]


def random_int_around(x, perc):
    r = -perc + (2 * perc * random.random())
    new_int = max(1, int(x * (1.0 + r / 100.00)))
    return new_int


def create_tasks(n_tasks: int, fat: float, regularity: float) -> Tuple[List[List[TaskInfo]], float]:
    # Compute the number of tasks per level
    n_tasks_per_level = int(fat * math.sqrt(n_tasks))
    total_tasks = 0
    total_comp_cost = 0.0
    tasks = []
    graph_comp_cost = random.uniform(0.0, 1.0)

    while total_tasks < n_tasks:
        n_tasks_at_level = min(
            random_int_around(n_tasks_per_level, 100.0 - 100.0 * regularity), n_tasks - total_tasks
        )
        comp_cost = round(random.uniform(0, 2 * graph_comp_cost), COST_PRECISION)
        total_comp_cost += comp_cost
        tasks_at_level = [
            TaskInfo(task_id, comp_cost, 0, 0, []) for task_id in range(
                total_tasks + 1, total_tasks + 1 + n_tasks_at_level
            )
        ]
        tasks.append(tasks_at_level)
        total_tasks += n_tasks_at_level

    return tasks, total_comp_cost


def create_dependencies(tasks: List[List[TaskInfo]], density: float, jump: int):
    n_levels = len(tasks)
    n_dependencies = 0

    # For all levels but the last one
    for level in range(1, n_levels):
        for level_idx in range(len(tasks[level])):
            # Compute how many parents the task should have
            n_tasks_upper_level = len(tasks[level - 1])
            n_parents = min(1 + int(random.uniform(0.0, density * n_tasks_upper_level)), n_tasks_upper_level)

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
                    n_dependencies += 1
    return n_dependencies


def create_communication_costs(total_comp_cost: float, ccr: float, num_edges: int) -> List[float]:
    total_comm_cost = total_comp_cost * ccr
    edge_costs = [random.random() for _ in range(num_edges)]

    # Adjust costs to match the total communication cost
    sum_edge_costs = sum(edge_costs)
    return [
        round(cost * total_comm_cost / sum_edge_costs, COST_PRECISION) for cost in edge_costs
    ]


def create_task_graph(task_info: List[List[TaskInfo]], comm_costs: List[float]) -> DiGraph:
    tasks, edges = [], []
    comm_idx = 0
    for sublist in task_info:
        for task in sublist:
            tasks.append((task.task_id, {"task_id": task.task_id, "processing_cost": task.computing_cost}))
            for dest in task.children:
                edges.append((task.task_id, dest.task_id, {"communication_cost": comm_costs[comm_idx]}))
                comm_idx += 1

    task_graph = DiGraph()
    task_graph.add_nodes_from(tasks)
    task_graph.add_edges_from(edges)
    return task_graph


def daggen(
        num_tasks: int = NUM_TASKS,
        fat: float = FAT,
        density: float = DENSITY,
        regularity: float = REGULARITY,
        jump: int = JUMP_SIZE,
        ccr: float = CCR
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

    tasks, total_comp_cost = create_tasks(n_tasks=num_tasks, fat=fat, regularity=regularity)
    num_edges = create_dependencies(tasks=tasks, density=density, jump=jump)
    comm_costs = create_communication_costs(total_comp_cost, ccr, num_edges)
    return create_task_graph(tasks, comm_costs)
