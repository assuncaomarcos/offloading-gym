#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from enum import IntEnum
from typing import List, Dict, Optional, Any, NamedTuple
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.utils import EzPickle
from gymnasium import error, spaces


__all__ = [
    "BaseOffEnv"
]


class OffloadingState(NamedTuple):
    task_graph: np.ndarray


class BaseOffEnv(ABC, gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    @abstractmethod
    def __init__(self, **kwargs):
        EzPickle.__init__(self)

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        pass
