#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List, Optional
from ..task_graph import TaskGraph


class WorkloadGenerator(ABC):
    @abstractmethod
    def step(self, offset: int = 1) -> List[Optional[TaskGraph]]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Returns the length of the workload. Zero if unbounded."""

    @abstractmethod
    def peek(self):
        """Peeks what would be the next task graph in the workload."""
