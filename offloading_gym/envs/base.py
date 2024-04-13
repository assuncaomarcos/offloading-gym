#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium.utils import EzPickle
from gymnasium.core import ObsType, ActType


class BaseOffEnv(ABC, gym.Env[ObsType, ActType], EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    @abstractmethod
    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self.renderer = kwargs.get("renderer", None)

    def render(self, mode='human'):
        if self.renderer is None:
            from offloading_gym.envs.render import OffloadingRenderer

            self.renderer = OffloadingRenderer(mode)
        rgb = self.renderer.render(self._render_state())
        return rgb

    def _render_state(self):
        ...

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError

    @property
    def reward(self):
        raise NotImplementedError
