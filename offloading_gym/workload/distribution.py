#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import Workload
from abc import ABC, abstractmethod
from typing import Optional, List
from ..task_graph import TaskGraph


class DistributionWorkload(Workload, ABC):
    """An abstract class for workload generation based on distributions.

        Attributes:
            length (int): An optional length of workload generation. The iteration will stop when length samples
                are generated.
            current_element (int): Stores the current element.

        Args:
            length (int, optional): Optional length of workload generation.
        """

    length: int
    current_element: int

    def __init__(self, length=0):
        self.length = length
        self.current_element = 0

    @abstractmethod
    def step(self, offset=1) -> List[Optional[TaskGraph]]:
        """Steps the workload generator by 'offset'.

        This may return new jobs, depending on the probability distributions of the workload.

        Args:
            offset (int): The number of time steps to advance.
        """
