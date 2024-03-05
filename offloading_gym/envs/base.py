#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
from typing import List, Dict, Optional, Any, NamedTuple
from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.utils import EzPickle
from gymnasium import error, spaces


__all__ = [
    "BaseOffEnv"
]


class OffloadingState(NamedTuple):
    task_graph: np.ndarray


class BaseOffEnv(ABC, gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array', 'ansi']}

    @abstractmethod
    def __init__(self, **kwargs):
        EzPickle.__init__(self)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

    def render(self, mode='human'):
        raise NotImplementedError

    @staticmethod
    def seed(seed=None):
        if seed is None:
            seed = random.randint(0, 99999999)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    @abstractmethod
    @property
    def state(self):
        raise NotImplementedError

    @property
    def reward(self):
        raise NotImplementedError

