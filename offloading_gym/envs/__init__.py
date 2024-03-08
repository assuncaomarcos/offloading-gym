#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Registers the gymnasium environments."""

from gymnasium.envs.registration import register

register(
    id="Offloading-v0",
    entry_point="offloading_gym.envs.offloading:OffloadingEnv",
    max_episode_steps=100
)
