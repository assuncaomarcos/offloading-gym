#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict
from collections.abc import Mapping, Set

from networkx import DiGraph
from networkx.classes.reportviews import NodeView
from functools import cached_property


__all__ = [
    'TaskAttr',
    'TaskGraph',
    'EdgeAttr'
]


class TaskAttr(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['task_id'] = kwargs.get('task_id', -1)
        self['task_size'] = kwargs.get('task_size', 0)
        self['processing_demand'] = kwargs.get('processing_demand', 0)
        self['output_datasize'] = kwargs.get('output_datasize', 0)

    @property
    def task_id(self) -> int:
        return self['task_id']

    @property
    def task_size(self) -> int:
        return self['task_size']

    @task_size.setter
    def task_size(self, value: int):
        self['task_size'] = value

    @property
    def processing_demand(self) -> int:
        return self['processing_demand']

    @processing_demand.setter
    def processing_demand(self, value: int):
        self['processing_demand'] = value

    @property
    def output_datasize(self) -> int:
        return self['output_datasize']

    @output_datasize.setter
    def output_datasize(self, value: int):
        self['output_datasize'] = value


class EdgeAttr(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['datasize'] = kwargs.get('datasize', 0)

    @property
    def datasize(self) -> int:
        return self['datasize']

    @datasize.setter
    def datasize(self, value: int):
        self['datasize'] = value


class TaskView(NodeView[TaskAttr], Mapping[Any, TaskAttr], Set[TaskAttr]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TaskGraph(DiGraph):
    _nodes: dict[int, TaskAttr]

    def __init__(self, **attr):
        super().__init__(**attr)

    @cached_property
    def tasks(self) -> TaskView:
        return TaskView(self)

    node_attr_dict_factory = TaskAttr
    edge_attr_dict_factory = EdgeAttr


