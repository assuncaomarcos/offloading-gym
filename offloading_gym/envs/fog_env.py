#!/usr/bin/env python
# -*- coding: utf-8 -*-

from offloading_gym.envs.base import BaseOffEnv
from typing import Optional, Any, Tuple, Dict
from numpy.typing import NDArray

import gymnasium as gym
import numpy as np

# class ServerEncoder(Callable[[TaskAttr], List[float]]):
#     cluster: Cluster
#
#     def __init__(self, cluster: Cluster):
#         self.cluster = cluster
#
#     def __call__(self, task: TaskAttr) -> List[float]:
#         local_exec_cost = self.cluster.local_execution_time(task.processing_demand)
#         upload_cost = self.cluster.upload_time(task.task_size)
#         edge_exec_cost = self.cluster.edge_execution_time(task.processing_demand)
#         download_cost = self.cluster.upload_time(task.output_datasize)
#         return [
#             float(task.task_id),
#             local_exec_cost,
#             upload_cost,
#             edge_exec_cost,
#             download_cost,
#         ]

GRID_COORDINATES = (0, )


class FogPlacementEnv(BaseOffEnv):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Dict

    def __init__(self, **kwargs):
        self._setup_spaces()

    def _setup_spaces(self):
        ...

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

    def step(self, action: int) -> Tuple[
        NDArray[np.float32],
        np.float32,
        bool,
        bool,
        Dict[str, Any],
    ]:
        ...

    @property
    def state(self):
        pass
