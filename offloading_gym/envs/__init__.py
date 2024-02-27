#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Registers the gym environments."""
from typing import Any

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec

# register(
#     id="CartPole-v0",
#     entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
#     vector_entry_point="gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv",
#     max_episode_steps=200,
#     reward_threshold=195.0,
# )
