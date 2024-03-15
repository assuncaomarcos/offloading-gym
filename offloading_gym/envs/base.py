#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium.utils import EzPickle
from gymnasium.core import ObsType, ActType


TASKS_PER_APPLICATION = 20


class BaseOffEnv(ABC, gym.Env[ObsType, ActType], EzPickle):
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

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError

    @property
    def reward(self):
        raise NotImplementedError
