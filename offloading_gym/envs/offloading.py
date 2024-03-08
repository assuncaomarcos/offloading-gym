#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gymnasium as gym
from .base import BaseOffEnv
from ..scheduler import Scheduler
from .workload import Workload
from typing import Union, Optional, Tuple, Any, NamedTuple
from .workload import DEFAULT_WORKLOAD_CONFIG, build_workload
import jax.numpy as jnp
import jax

# Number of downstream/upstream tasks considered when encoding a task graph
SUCCESSOR_TASKS = PREDECESSOR_TASKS = 6
TASK_PROFILE_LENGTH = 5


class OffloadingState(NamedTuple):
    task_graph: jax.Array
    task_dependency: jax.Array
    scheduling_plan: jax.Array


class OffloadingEnv(BaseOffEnv):
    use_raw_state: bool
    observation_space: Union[gym.spaces.tuple.Tuple, gym.spaces.box.Box]
    action_space: gym.spaces.MultiBinary
    scheduler: Scheduler
    workload: Workload

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_raw_state = kwargs.get("use_raw_state", True)
        self._setup_spaces()

        self.workload_config = kwargs.get('workload', DEFAULT_WORKLOAD_CONFIG)
        self.workload = build_workload(self.workload_config)

    def _setup_spaces(self):
        self.action_space = gym.spaces.MultiBinary(self.tasks_per_app)
        self.setup_raw_state() if self.use_raw_state else self.setup_image_state()

    def setup_raw_state(self):
        task_graph_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(
                self.tasks_per_app,
                TASK_PROFILE_LENGTH
            )
        )

        task_dependency_space = gym.spaces.Box(
            low=0, high=1.0,
            shape=(
                self.tasks_per_app,
                SUCCESSOR_TASKS + PREDECESSOR_TASKS
            ),
        )

        scheduling_plan_space = gym.spaces.MultiBinary(self.tasks_per_app)

        self.observation_space = gym.spaces.Tuple(
            (task_graph_space, task_dependency_space, scheduling_plan_space)
        )

    def setup_image_state(self):
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                self.tasks_per_app,
                TASK_PROFILE_LENGTH + SUCCESSOR_TASKS + PREDECESSOR_TASKS,
            ),
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[OffloadingState, dict[str, Any]]:
        super().reset(seed=seed)
        return self.state, {}

    def step(self, action: int) -> Tuple[OffloadingState, dict[str, Any]]:
        pass

    # def _get_obs(self):
    #     return {"agent": self._agent_location, "target": self._target_location}

    @property
    def state(self):
        pass
        # return OffloadingState(
        #     task_graph=self.tasks_per_app, TASK_PROFILE_LENGTH))
        # )

