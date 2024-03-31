#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module generates Directed Acyclic Graphs (DAGs) based on parameters such as task number,
networking features, and Communication to Computation Ratio (CCR). This is useful for
simulating task scheduling in parallel and distributed systems.

The module is based on the daggen random graph generator proposed by Suter & Hunold,
with modifications by Arabnejad and Barbosa. For more details, see the article below:

H. Arabnejad and J. Barbosa, 'List Scheduling Algorithm for Heterogeneous Systems by
an Optimistic Cost Table', IEEE Transactions on Parallel and Distributed Systems,
Vol. 25, No. 3, March 2014.

This module utilizes NumPy's random number generator to produce random numbers.
To ensure reproducibility, you can provide a specific random number generator when
generating a random DAG. If none is provided, a module-level random number generator,
created with a random seed, will be used.

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
from typing import List, Any, Optional
import numpy as np

__all__ = ["random_dag"]


# Default values for generated DAGs
NUM_TASKS = 20
FAT = 0.7
REGULARITY = 0.5
DENSITY = 0.6
JUMP_SIZE = 1
CCR = 0.3
MIN_DATA = 5120  # Data sizes of 5KB - 50KB
MAX_DATA = 51200
MIN_COMPUTATION = 10**7  # Each task requires between 10^7 and 10^8 cycles
MAX_COMPUTATION = 10**8

COST_PRECISION = 4


@dataclass
class TaskInfo:
    task_id: int
    computing_cost: float
    data_cost: float
    n_children: int
    children: List[Any]


_rng = np.random.default_rng()


def random_int_in_range(
    rng: np.random.Generator, num: float, range_percent: float
) -> int:
    """Generates a random integer within a range around a specified number."""
    r = -range_percent + (2 * range_percent * rng.random())
    return max(1, int(num * (1.0 + r / 100.00)))


def scale_array(
    input_array: List[float], min_val: float, max_val: float
) -> List[float]:
    np_array = np.array(input_array)
    min_arr = np.amin(np_array)
    max_arr = np.amax(np_array)
    output_array = (np_array - min_arr) / (max_arr - min_arr)
    return list(output_array * (max_val - min_val) + min_val)


def create_tasks(
    rng: np.random.Generator,
    n_tasks: int,
    fat: float,
    regularity: float,
    min_comp: int,
    max_comp: int,
) -> List[List[TaskInfo]]:
    # Compute the number of tasks per level
    n_tasks_per_level = int(fat * math.sqrt(n_tasks))
    total_tasks = 0
    tasks = []

    while total_tasks < n_tasks:
        n_tasks_at_level = min(
            random_int_in_range(rng, n_tasks_per_level, 100.0 - 100.0 * regularity),
            n_tasks - total_tasks,
        )
        tasks_at_level = [
            TaskInfo(
                task_id=task_id,
                computing_cost=rng.integers(low=min_comp, high=max_comp, endpoint=True),
                data_cost=0,
                n_children=0,
                children=[],
            )
            for task_id in range(total_tasks + 1, total_tasks + 1 + n_tasks_at_level)
        ]
        tasks.append(tasks_at_level)
        total_tasks += n_tasks_at_level

    return tasks


def create_dependencies(
    rng: np.random.Generator, tasks: List[List[TaskInfo]], density: float, jump: int
) -> None:
    n_levels = len(tasks)

    # For all levels but the last one
    for level in range(1, n_levels):
        for level_idx in range(len(tasks[level])):
            # Compute how many parents the task should have
            n_tasks_upper_level = len(tasks[level - 1])
            n_parents = min(
                1 + int(rng.uniform(0.0, density * n_tasks_upper_level)),
                n_tasks_upper_level,
            )

            for _ in range(n_parents):
                # compute the level of the parent
                parent_level = max(0, level - int(rng.uniform(1.0, jump + 1)))
                parent_idx = int(rng.uniform(0, len(tasks[parent_level])))
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
    tasks: List[List[TaskInfo]], ccr: float, min_data: int, max_data: int
) -> None:
    # To compute the CCR the original model has communication costs associated to edges,
    # but to reproduce the results of the MRLCO paper, we consider all tasks here since the result of
    # sinks may need to be returned to the local device.
    flat_task_list = [task for level in tasks for task in level]
    comp_costs = [task.computing_cost * ccr for task in flat_task_list]
    comm_costs = scale_array(input_array=comp_costs, min_val=min_data, max_val=max_data)

    for task, cost in zip(flat_task_list, comm_costs):
        task.data_cost = int(cost)


def create_task_graph(task_info: List[List[TaskInfo]]) -> DiGraph:
    tasks, edges = [], []
    for sublist in task_info:
        for task in sublist:
            tasks.append(
                (
                    task.task_id,
                    {
                        "task_id": task.task_id,
                        "processing_demand": task.computing_cost,
                        "output_datasize": task.data_cost,
                    },
                )
            )
            for dest in task.children:
                edges.append(
                    (
                        task.task_id,
                        dest.task_id,
                        {"datasize": task.data_cost},
                    )
                )

    task_graph = DiGraph()
    task_graph.add_nodes_from(tasks)
    task_graph.add_edges_from(edges)
    return task_graph


def random_dag(
    *,
    rng: Optional[np.random.Generator] = None,
    num_tasks: Optional[int] = NUM_TASKS,
    fat: Optional[float] = FAT,
    density: Optional[float] = DENSITY,
    regularity: Optional[float] = REGULARITY,
    jump: Optional[int] = JUMP_SIZE,
    ccr: Optional[float] = CCR,
    min_comp: Optional[int] = MIN_COMPUTATION,
    max_comp: Optional[int] = MAX_COMPUTATION,
    min_data: Optional[int] = MIN_DATA,
    max_data: Optional[int] = MAX_DATA,
) -> DiGraph:
    """Generates a random task DAG.

    Args:
        rng: the Numpy random number generator to use.
        num_tasks: the number of tasks in the DAG. Default is 20.
        fat: the fatness factor of the DAG. Default is 0.7.
        density: the density factor of the dependencies between tasks. Default is 0.6.
        regularity: the regularity factor of the DAG. Default is 0.5.
        jump: the jump size for creating dependencies between tasks. Default is 1.
        ccr: the communication-to-computation ratio. Default is 0.3.
        min_comp: the minimum computation cost for tasks.
        max_comp: the maximum computation cost for tasks.
        min_data: the minimum data size for tasks' communication costs.
        max_data: the maximum data size for tasks' communication costs.

    Returns:
        A DiGraph representing the randomly generated DAG with the specified parameters.

    Raises:
        AssertionError: If any of the input parameters do not meet their constraints.

    """
    assert num_tasks >= 1, "num_tasks must be >= 1"
    assert 0 <= fat, "fat must be greater than or equal to 0"
    assert 0 <= density <= 1, "density must be between 0 and 1"
    assert 0 <= regularity <= 1, "regularity must be between 0 and 1"
    assert 0 <= ccr <= 10, "ccr must be between 0 and 10"
    assert 1 <= jump <= 4, "jump_size must be between 1 and 4"
    assert (
        0 <= min_comp <= max_comp
    ), "min_comp must be smaller than max_comp, and both must be greater than 0"
    assert (
        0 <= min_data <= max_data
    ), "min_data must be smaller than max_data, and both must be greater than 0"

    if rng is None:
        rng = _rng

    tasks = create_tasks(
        rng=rng,
        n_tasks=num_tasks,
        fat=fat,
        regularity=regularity,
        min_comp=min_comp,
        max_comp=max_comp,
    )
    create_dependencies(rng, tasks, density, jump)
    set_communication_costs(tasks, ccr, min_data, max_data)
    task_graph = create_task_graph(tasks)
    return task_graph
