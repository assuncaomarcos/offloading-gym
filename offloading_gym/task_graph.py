#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Any
from collections.abc import Mapping, Set

import networkx as nx
from networkx import DiGraph
from networkx.classes.reportviews import NodeView
from functools import cached_property

__all__ = [
    'Task',
    'TaskGraph'
]


class Task(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['id'] = None
        self['processing_demand'] = None
        self['input_bytes'] = None
        self['output_bytes'] = None

    @property
    def id(self) -> int:
        return self['id']

    @id.setter
    def id(self, value: int):
        self['id'] = value

    @property
    def processing_demand(self) -> int:
        return self['processing_demand']

    @processing_demand.setter
    def processing_demand(self, value: int):
        self['processing_demand'] = value

    @property
    def input_bytes(self) -> int:
        return self['input_bytes']

    @input_bytes.setter
    def input_bytes(self, value: int):
        self['input_bytes'] = value

    @property
    def output_bytes(self) -> int:
        return self["output_bytes"]

    @output_bytes.setter
    def output_bytes(self, value: int):
        self['output_bytes'] = value


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
