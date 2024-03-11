#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import jax as jnp
import numpy as np
from typing import Tuple, Optional, Any, NamedTuple
from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.utils import EzPickle


TASKS_PER_APPLICATION = 20

__all__ = [
    "BaseOffEnv"
]


class BaseOffEnv(ABC, gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    tasks_per_app: int

    @abstractmethod
    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self.tasks_per_app = kwargs.get("tasks_per_app", TASKS_PER_APPLICATION)
        self.renderer = kwargs.get("renderer", None)

    def render(self, mode='human'):
        if self.renderer is None:
            from .render import OffloadingRenderer

            self.renderer = OffloadingRenderer(mode)
        rgb = self.renderer.render(self._render_state())
        return rgb

    def _render_state(self):
        pass

    @staticmethod
    def seed(seed=None):
        if seed is None:
            seed = random.randint(0, 99999999)
        jnp.random.seed(seed)
        np.random.seed(seed)      # In case some other library uses numpy
        random.seed(seed)
        return [seed]

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError

    @property
    def reward(self):
        raise NotImplementedError

