#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Any
from collections.abc import Mapping, Set

import daggen
import random
import math
from networkx import DiGraph
from networkx.classes.reportviews import NodeView
from functools import cached_property
from networkx.drawing.nx_agraph import from_agraph


__all__ = [
    'Task',
    'TaskGraph',
    'parse_dot',
    'daggen_graph'
]

# Default values for daggen DAGs
NUM_TASKS = 20
FAT = 0.5
REGULAR = 0.5
DENSITY = 0.6
MIN_DATA = 2048
MAX_DATA = 11264
MIN_ALPHA = 0.0
MAX_ALPHA = 0.2
JUMP_SIZE = 1
CCR = 0


class Task(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['id'] = kwargs.get('id', -1)
        self['processing'] = kwargs.get('processing', 0)
        self['data_size'] = kwargs.get('data_size', 0)
        self['output_size'] = kwargs.get('output_size', 0)

    @property
    def id(self) -> int:
        return self['id']

    @id.setter
    def id(self, value: int):
        self['id'] = value

    @property
    def processing_size(self) -> int:
        return self['processing_size']

    @processing_size.setter
    def processing_size(self, value: int):
        self['processing_size'] = value

    @property
    def data_size(self) -> int:
        return self['data_size']

    @data_size.setter
    def data_size(self, value: int):
        self['data_size'] = value

    @property
    def output_size(self) -> int:
        return self["output_size"]

    @output_size.setter
    def output_size(self, value: int):
        self['output_size'] = value


class TaskView(NodeView[Task], Mapping[Any, Task], Set[Task]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TaskGraph(DiGraph):
    _nodes: dict[int, Task]

    def __init__(self, **attr):
        super().__init__(**attr)

    @cached_property
    def tasks(self) -> TaskView:
        return TaskView(self)

    node_dict_factory = dict
    node_attr_dict_factory = Task
    edge_attr_dict_factory = dict


def parse_dot(path: str) -> TaskGraph:
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError(
            "parse() requires pygraphviz " "https://pygraphviz.github.io"
        ) from err

    gr = pygraphviz.AGraph(file=path)
    tg = from_agraph(gr, create_using=TaskGraph)
    gr.clear()
    return tg


def daggen_graph(
        num_tasks: int = NUM_TASKS,
        min_data: int = MIN_DATA,
        max_data: int = MAX_DATA,
        min_alpha: float = MIN_ALPHA,
        max_alpha: float = MAX_ALPHA,
        fat: float = FAT,
        density: float = DENSITY,
        regular: float = REGULAR,
        ccr: int = CCR
) -> TaskGraph:
    assert num_tasks >= 1, "num_tasks must be >= 1"
    assert 0 <= min_data <= max_data, "min_data and max_data must be >= 0, and min_data must be <= max_data"
    assert 0 <= min_alpha <= max_alpha <= 1, ("min_alpha and max_alpha must be between 0..1, "
                                              "min_alpha must be smaller than max_alpha")
    assert 0 <= fat <= 1, "fat must be between 0 and 1"
    assert 0 <= density <= 1, "density must be between 0 and 1"
    assert 0 <= regular <= 1, "regular must be between 0 and 1"
    assert 0 <= ccr <= 3, "ccr must be between 0 and 3"

    random_int = random.randint(1, 9999999)
    dag = daggen.DAG(
        seed=random_int, num_tasks=num_tasks,
        min_data=min_data, max_data=max_data,
        min_alpha=min_alpha, max_alpha=max_alpha,
        fat=fat, density=density, regular=regular, ccr=ccr
    )
    task_tuples, edge_tuples = dag.task_n_edge_tuples()
    task_graph = TaskGraph()

    tasks = []
    for task_id, task_data in task_tuples:
        tasks.append((task_id, {'id': task_id, 'processing': task_data['computation']}))

    for source, _, data in edge_tuples:
        tasks[source - 1][1]['data_size'] = int(data['data'] / 8.0)

    for _, task_data in tasks:
        if 'data_size' not in task_data:
            # Use the same approach of daggen to create the input size of other tasks
            task_data['data_size'] = int(math.pow(random.uniform(min_data, max_data), 2.0))

    task_graph.add_nodes_from(tasks)
    task_graph.add_edges_from(edge_tuples)

    # print(task_tuples)
    # print(edge_tuples)
    #
    for task in task_graph.tasks.values():
        print(task)
        # print(task)

    return task_graph
