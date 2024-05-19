#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gymnasium as gym

from abc import ABC, abstractmethod
from typing import Tuple

__all__ = ['DRLModel']


class DRLModel(ABC):
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]

    def __init__(self, env: gym.Env):
        self.input_shape = env.observation_space.shape
        self.output_shape = env.action_space.shape

    @abstractmethod
    def select_actions(self, state):
        ...

    @abstractmethod
    def process_experiences(self, experiences):
        ...
