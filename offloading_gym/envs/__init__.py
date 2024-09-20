#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Registers the gymnasium environments."""

from gymnasium.envs.registration import register
from .base import BaseOffEnv
from .binary import BinaryOffloadEnv
from .fog import FogPlacementEnv

register(
    id="BinaryOffload-v0",
    entry_point="offloading_gym.envs.binary:BinaryOffloadEnv",
    max_episode_steps=100,
)

register(
    id="FogPlacement-v0",
    entry_point="offloading_gym.envs.fog:FogPlacementEnv",
    max_episode_steps=100,
)
