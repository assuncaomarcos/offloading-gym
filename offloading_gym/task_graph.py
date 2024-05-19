#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A module for extending NetworkX's DiGraph to handle task graphs.

This module provides helper classes that extend the DiGraph functionalities
of NetworkX to support the specific requirements of handling task graphs.
Task graphs are used in the scheduling simulations of environments
and workloads.

Classes:
    TaskAttr: Represents the attributes (dict) of an individual task.
    TaskGraph: Enhances NetworkX's DiGraph to manage tasks.
    EdgeAttr: Represents the attributes (dict) of an individual edge.
    TaskTuple: Tuple containing a task id and a TaskAttr with task attributes.
    EdgeTuple: Tuple containing an edge id and a EdgeAttr with edge attributes.
"""

from typing import Any, Tuple
from collections.abc import Mapping, Set

from functools import cached_property
from networkx import DiGraph
from networkx.classes.reportviews import NodeView


__all__ = [
    'TaskAttr',
    'TaskGraph',
    'EdgeAttr',
    'EdgeTuple',
    'TaskTuple'
]


class TaskAttr(dict):
    """ Represents the attributes (dict) of an individual task."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['task_id'] = kwargs.get('task_id', -1)
        self['task_size'] = kwargs.get('task_size', 0)
        self['processing_demand'] = kwargs.get('processing_demand', 0)
        self['output_datasize'] = kwargs.get('output_datasize', 0)

    @property
    def task_id(self) -> int:
        """The task id."""
        return self['task_id']

    @property
    def task_size(self) -> int:
        """The size of the task's input data/code in bytes."""
        return self['task_size']

    @task_size.setter
    def task_size(self, value: int):
        """Set the size of the task's input data/code in bytes."""
        self['task_size'] = value

    @property
    def processing_demand(self) -> int:
        """The processing demand of the task in number of CPU cycles."""
        return self['processing_demand']

    @processing_demand.setter
    def processing_demand(self, value: int):
        """Set the processing demand of the task in number of CPU cycles."""
        self['processing_demand'] = value

    @property
    def output_datasize(self) -> int:
        """The output data size of the task in bytes."""
        return self['output_datasize']

    @output_datasize.setter
    def output_datasize(self, value: int):
        """Set the output data size of the task in bytes."""
        self['output_datasize'] = value


class EdgeAttr(dict):
    """ Represents the attributes (dict) of an individual edge."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['datasize'] = kwargs.get('datasize', 0)

    @property
    def datasize(self) -> int:
        """The amount of data in bytes sent from an upstream task to a downstream task"""
        return self['datasize']

    @datasize.setter
    def datasize(self, value: int):
        """Set the amount of data in bytes sent from an upstream task to a downstream task"""
        self['datasize'] = value


class TaskView(NodeView[TaskAttr], Mapping[Any, TaskAttr], Set[TaskAttr]):
    """Helper node view for a NetworkX DiGraph."""


class TaskGraph(DiGraph):
    """ Represents a task graph as a NetworkX DiGraph."""
    _nodes: dict[int, TaskAttr]

    def __init__(self, **attr):
        super().__init__(**attr)

    @cached_property
    def tasks(self) -> TaskView:
        """The tasks in the task graph"""
        return TaskView(self)

    node_attr_dict_factory = TaskAttr
    edge_attr_dict_factory = EdgeAttr


TaskTuple = Tuple[int, TaskAttr]
EdgeTuple = Tuple[int, EdgeAttr]
