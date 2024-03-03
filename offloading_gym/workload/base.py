#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List, Optional
from ..task_graph import TaskGraph


class Workload(ABC):
    """An abstract class for workload generators.

    Attributes:
        current_time: The current time in a unit used in the simulation
        length (int): Optional length of workload generation.
        current_element (int): Stores the current element.
    """

    current_time: int
    length: int
    current_element: int

    @abstractmethod
    def __init__(self, length=0):
        self.length = length
        self.current_element = 0
        self.current_time = 0

    @abstractmethod
    def step(self, offset: int = 1) -> List[Optional[TaskGraph]]:
        """Steps the workload generator by 'offset'"""

    @abstractmethod
    def __len__(self):
        """Returns the length of the workload. Zero if unbounded."""

    @abstractmethod
    def peek(self):
        """Peeks what would be the next task graph in the workload."""
