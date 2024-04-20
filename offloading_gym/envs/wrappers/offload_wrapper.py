#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gymnasium as gym
from gymnasium import spaces

import numpy as np


class MultiBinaryToDiscreteWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiBinary)
        self.n = env.action_space.n
        self.action_space = gym.spaces.Discrete(2 ** self.n)

    def action(self, action):
        bits = "{0:b}".format(action)
        bits = '0' * (self.n - len(bits)) + bits
        return np.array([int(b) for b in bits], dtype=np.int64)


class ReshapeActionWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiBinary)
        self.n = env.action_space.n
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n, 1), dtype=np.int32)

    def action(self, action: np.ndarray):
        return action.reshape(-1)
