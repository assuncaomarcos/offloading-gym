#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Registers the gymnasium environments."""

from gymnasium.envs.registration import register

register(
    id="BinaryOffload-v0",
    entry_point="offloading_gym.envs.drlto.offload_env:BinaryOffloadEnv",
    max_episode_steps=100
)
