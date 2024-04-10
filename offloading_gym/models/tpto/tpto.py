#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gymnasium as gym

from offloading_gym.models.base import DRLModel


class TransformerPPO(DRLModel):

    def __init__(self, env: gym.Env):
        super().__init__(env=env)

    def select_actions(self, state):
        pass

    def process_experiences(self, experiences):
        ...
