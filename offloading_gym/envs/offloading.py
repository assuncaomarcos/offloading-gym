#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple, List, Any, Callable, SupportsFloat
from numpy.typing import NDArray

import gymnasium as gym
import numpy as np

from .base import BaseOffEnv
from .scheduler import build_scheduler, DEFAULT_SCHEDULER_CONFIG
from .workload import build_workload, RANDOM_WORKLOAD_CONFIG
from ..cluster import Cluster
from ..scheduler import Scheduler
from ..task_graph import TaskGraph, TaskAttr
from ..utils import arrays
from ..workload import Workload

__all__ = [
    'OffloadingEnv'
]

# Number of downstream/upstream tasks considered when encoding a task graph
SUCCESSOR_TASKS = PREDECESSOR_TASKS = 6
TASK_PROFILE_LENGTH = 5

# Columns of the graph embedding that contain task ids and times
EMBED_ID_COLUMNS = [0] + list(range(TASK_PROFILE_LENGTH, TASK_PROFILE_LENGTH + PREDECESSOR_TASKS + SUCCESSOR_TASKS))
EMBED_TIME_COLUMNS = list(range(1, TASK_PROFILE_LENGTH))


class TaskCostEncoder(Callable[[TaskAttr], List[float]]):
    cluster: Cluster

    def __init__(self, cluster: Cluster):
        self.cluster = cluster

    def __call__(self, task: TaskAttr) -> List[float]:
        local_exec_cost = self.cluster.local_execution_time(task.processing_demand)
        upload_cost = self.cluster.transmission_time(task.task_size)
        edge_exec_cost = self.cluster.edge_execution_time(task.processing_demand)
        download_cost = self.cluster.transmission_time(task.output_datasize)
        return [float(task.task_id), local_exec_cost, upload_cost, edge_exec_cost, download_cost]


class OffloadingEnv(BaseOffEnv):
    use_raw_state: bool
    scheduler: Scheduler
    workload: Workload
    task_graph_space: Optional[gym.spaces.Box]
    scheduling_plan_space: Optional[gym.spaces.MultiBinary]

    # Raw observation space = task_graph + scheduling plan
    observation_space: Union[gym.spaces.tuple.Tuple, gym.spaces.box.Box]
    action_space: gym.spaces.MultiBinary
    graph_embedding: Union[NDArray, None]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_raw_state = kwargs.get("use_raw_state", True)
        self._setup_spaces()

        self.workload_config = kwargs.get('workload', RANDOM_WORKLOAD_CONFIG)
        self.workload = build_workload(self.workload_config)
        self.scheduler_config = kwargs.get('scheduler', DEFAULT_SCHEDULER_CONFIG)
        self.scheduler = build_scheduler(self.scheduler_config)
        self.task_encoder = TaskCostEncoder(self.scheduler.cluster)
        self.task_graph = None
        self.graph_embedding = None

    def _setup_spaces(self):
        self.action_space = gym.spaces.MultiBinary(self.tasks_per_app)
        self.setup_raw_state() if self.use_raw_state else self.setup_image_state()

    def setup_raw_state(self):
        self.task_graph_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(
                self.tasks_per_app,
                TASK_PROFILE_LENGTH + SUCCESSOR_TASKS + PREDECESSOR_TASKS
            )
        )

        self.scheduling_plan_space = gym.spaces.MultiBinary(self.tasks_per_app)

        self.observation_space = gym.spaces.Tuple(
            (self.task_graph_space, self.scheduling_plan_space)
        )

    def setup_image_state(self):
        self.observation_space = gym.spaces.Box(
            low=-1.0,
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
    ) -> Tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.workload.reset(seed=seed)

        # Use only the first task graph
        self.task_graph = self.workload.step(offset=0)[0]
        compute_task_ranks(self.scheduler.cluster, self.task_graph)
        self.graph_embedding = task_embeddings(task_graph=self.task_graph, task_encoder=self.task_encoder)
        return self._get_ob(), {}

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using OffloadingEnv."
        return self.graph_embedding

    def step(
            self,
            action: NDArray[np.int8]
    ) -> Tuple[NDArray[np.float32], SupportsFloat, bool, bool, dict[str, Any]]:
        pass

    @property
    def state(self) -> NDArray[np.float32]:
        return self.graph_embedding


def compute_task_ranks(cluster: Cluster, task_graph: TaskGraph):
    """ Computes the task ranks as per the MRLCO paper. """
    successors = task_graph.succ

    def task_runtime(task: TaskAttr) -> float:
        local_exec = cluster.local_execution_time(task.processing_demand)
        upload = cluster.transmission_time(task.task_size)
        edge_exec = cluster.edge_execution_time(task.processing_demand)
        download = cluster.transmission_time(task.output_datasize)
        return min(local_exec, upload + edge_exec + download)

    def task_rank(task: TaskAttr) -> float:
        rank = task.get("rank")
        if rank is not None:
            return rank

        runtime = task["estimated_runtime"]
        task_successors = successors[task.task_id]
        if len(task_successors) == 0:
            task["rank"] = runtime
            return runtime
        else:
            task["rank"] = runtime + max(task_rank(task_graph.nodes[j]) for j in task_successors.keys())
            return task["rank"]

    for task_attr in task_graph.nodes().values():
        task_attr["estimated_runtime"] = task_runtime(task_attr)

    for task_attr in task_graph.nodes().values():
        task_attr["rank"] = task_rank(task_attr)


def task_embeddings(task_graph: TaskGraph, task_encoder: Callable[[TaskAttr], List[float]]) -> np.ndarray:
    """ Creates a list of task embeddings as per the MRLCO paper. """
    task_info = []
    for task_id, task_attr in task_graph.nodes().items():
        task_predecessors = list(task_graph.pred[task_id].keys())
        task_successors = list(task_graph.succ[task_id].keys())
        encoded_task = task_encoder(task_attr)
        task_info.append(
            encoded_task
            + arrays.pad_list(lst=task_predecessors, target_length=PREDECESSOR_TASKS, pad_value=-1.0)
            + arrays.pad_list(lst=task_successors, target_length=SUCCESSOR_TASKS, pad_value=-1.0)
        )

    def normalize_task_ids(dependencies: np.ndarray) -> np.ndarray:
        mask = dependencies != -1
        min_val = dependencies[mask].min()
        max_val = dependencies[mask].max()
        dependencies[mask] = (dependencies[mask] - min_val) / (max_val - min_val)
        return dependencies

    def normalize_times(features: np.ndarray) -> np.ndarray:
        return (features - features.min()) / (features.max() - features.min())

    embeddings = np.array(task_info, dtype=np.float32)
    embeddings[:, EMBED_ID_COLUMNS] = normalize_task_ids(embeddings[:, EMBED_ID_COLUMNS])
    embeddings[:, EMBED_TIME_COLUMNS] = normalize_times(embeddings[:, EMBED_TIME_COLUMNS])

    return embeddings
