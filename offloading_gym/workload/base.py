#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import List, Optional, Union
from ..task_graph import TaskGraph, TaskAttr
import numpy as np
from gymnasium.utils import seeding


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
    _np_random: Union[np.random.Generator, None] = None

    task_attr_factory = TaskAttr

    @abstractmethod
    def __init__(self, length: Optional[int] = 0):
        self.length = length
        self.current_element = 0
        self.current_time = 0

    @abstractmethod
    def step(self, *, offset: Optional[int] = 1) -> List[Optional[TaskGraph]]:
        """Steps the workload generator by 'offset'"""

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Returns the workload to its original state."""
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.current_element = 0
        self.current_time = 0

    @abstractmethod
    def __len__(self):
        """Returns the length of the workload. Zero if unbounded."""

    @abstractmethod
    def peek(self):
        """Peeks what would be the next task graph in the workload."""

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal _np_random attribute.
        If not already set, this attribute will be initialised with a random seed.

        Returns:
            Generator: An instance of `np.random.Generator`.
        """
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value
