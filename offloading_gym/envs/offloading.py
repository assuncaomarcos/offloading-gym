#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Optional, Tuple, List, Any, Callable, SupportsFloat, Dict
from numpy.typing import NDArray
from functools import cached_property

import gymnasium as gym
import numpy as np
import networkx as nx

from .base import BaseOffEnv
from .workload import build_workload, RANDOM_WORKLOAD_CONFIG
from ..task_graph import TaskGraph, TaskAttr, TaskTuple
from ..utils import arrays
from ..workload import Workload
from ..simulator import Cluster, Simulator, TaskExecution


__all__ = [
    'OffloadingEnv'
]

TASKS_PER_APPLICATION = 20

# Number of embedding fields containing task properties
TASK_PROFILE_LENGTH = 5

# Number of downstream/upstream tasks considered when encoding a task graph
TASK_SUCCESSORS = TASK_PREDECESSORS = 6

# Columns of the graph embedding that contain task tims and IDs
TASK_TIME_COLUMNS = list(range(1, TASK_PROFILE_LENGTH))
TASK_ID_COLUMNS = [0] + list(
    range(
        TASK_PROFILE_LENGTH,
        TASK_PROFILE_LENGTH + TASK_PREDECESSORS + TASK_SUCCESSORS
    )
)


DEFAULT_CLUSTER_CONFIG = {
    "num_edge_cpus": 10,
    "edge_cpu_capacity": 1024 * 1024,
    "num_local_cpus": 1,
    "local_cpu_capacity": 1024 * 1024,
    "upload_rate": 1,
    "download_rate": 1
}


class TaskCostEncoder(Callable[[TaskAttr], List[float]]):
    cluster: Cluster

    def __init__(self, cluster: Cluster):
        self.cluster = cluster

    def __call__(self, task: TaskAttr) -> List[float]:
        local_exec_cost = self.cluster.local_execution_time(task.processing_demand)
        upload_cost = self.cluster.upload_time(task.task_size)
        edge_exec_cost = self.cluster.edge_execution_time(task.processing_demand)
        download_cost = self.cluster.upload_time(task.output_datasize)
        return [float(task.task_id), local_exec_cost, upload_cost, edge_exec_cost, download_cost]


class OffloadingEnv(BaseOffEnv):
    use_raw_state: bool
    workload: Workload
    cluster: Cluster
    tasks_per_app: int

    observation_space: gym.spaces.Tuple
    action_space: gym.spaces.MultiBinary

    # Current task graph
    task_graph: Union[TaskGraph, None]

    # A numpy array containing the encoded task graph
    scheduling_plan: NDArray[np.int8]
    graph_embedding: Union[NDArray, None]

    # Tasks sorted for scheduling for avoiding having to sort them multiple times
    task_list: Union[List[TaskTuple], None]

    # To truncate episodes
    max_episode_steps: Union[int, None]
    steps: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks_per_app = kwargs.get("tasks_per_app", TASKS_PER_APPLICATION)
        self.max_episode_steps = kwargs.get("max_episode_steps", None)
        self._setup_spaces()
        self._build_simulation_elements(kwargs)
        self.task_graph = None
        self.graph_embedding = None
        self.task_list = None

    def _build_simulation_elements(self, kwargs):
        workload_config = kwargs.get('workload', RANDOM_WORKLOAD_CONFIG)
        cluster_config = kwargs.get('cluster', DEFAULT_CLUSTER_CONFIG)
        self.cluster = Cluster(**cluster_config)
        self.workload = build_workload(workload_config)
        self.task_encoder = TaskCostEncoder(self.cluster)

    def _setup_spaces(self):
        self.action_space = gym.spaces.MultiBinary(self.tasks_per_app)

        scheduling_plan_space = gym.spaces.MultiBinary(self.tasks_per_app)
        task_embedding_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(
                self.tasks_per_app,
                TASK_PROFILE_LENGTH + TASK_SUCCESSORS + TASK_PREDECESSORS,
            ),
        )
        self.observation_space = gym.spaces.Tuple((scheduling_plan_space, task_embedding_space))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Tuple[NDArray[np.int8], NDArray[np.float32]], dict[str, Any]]:
        super().reset(seed=seed)
        self.workload.reset(seed=seed)
        self.steps = 0

        # Use only the first task graph
        self.task_graph = self.workload.step(offset=0)[0]
        self.compute_task_ranks(self.cluster, self.task_graph)
        self.scheduling_plan = np.zeros(shape=(self.tasks_per_app,), dtype=np.int8)

        topo_order = nx.lexicographical_topological_sort(
            self.task_graph, key=lambda task_id: self.task_graph.nodes[task_id]["rank"]
        )

        # Store sorted tasks to avoid having to sort it multiple times
        self.task_list = [(node, self.task_graph.nodes[node]) for node in topo_order]

        self.graph_embedding = self.task_embeddings(
            task_graph=self.task_graph,
            sorted_tasks=self.task_list,
            task_encoder=self.task_encoder
        )

        return self._get_ob(), {}

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using OffloadingEnv."
        return self.scheduling_plan, self.graph_embedding

    def step(
            self,
            action: NDArray[np.int8]
    ) -> Tuple[Tuple[NDArray[np.int8], NDArray[np.float32]], SupportsFloat, bool, bool, Dict[str, Any]]:
        self.scheduling_plan = action
        self.steps += 1
        action_execution = Simulator.build(self.cluster).simulate(
            self.task_list,
            action.tolist()
        )
        local_execution = Simulator.build(self.cluster).simulate(
            self.task_list,
            self.all_local_action
        )
        scores = self._latency_scores(action_execution, local_execution)
        truncate = self.max_episode_steps is not None and self.steps >= self.max_episode_steps
        return self._get_ob(), np.mean(scores), False, truncate, {}

    def _latency_scores(
            self,
            action_execution: List[TaskExecution],
            local_execution: List[TaskExecution]
    ) -> NDArray[np.float32]:
        ms_plan = np.array([task_execution.make_span for task_execution in action_execution])
        ms_local = np.array([task_execution.make_span for task_execution in local_execution])
        avg_ms_local = ms_local / float(self.tasks_per_app)
        scores = -(ms_plan - avg_ms_local) / ms_local
        return scores

    @cached_property
    def all_local_action(self):
        return [0] * self.tasks_per_app

    @property
    def state(self) -> Tuple[NDArray[np.int8], NDArray[np.float32]]:
        return self.scheduling_plan, self.graph_embedding

    @staticmethod
    def compute_task_ranks(cluster: Cluster, task_graph: TaskGraph):
        """ Computes the task ranks as per the MRLCO paper. """
        successors = task_graph.succ

        def task_runtime(task: TaskAttr) -> float:
            local_exec = cluster.local_execution_time(task.processing_demand)
            upload = cluster.upload_time(task.task_size)
            edge_exec = cluster.edge_execution_time(task.processing_demand)
            download = cluster.download_time(task.output_datasize)
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

    @staticmethod
    def task_embeddings(
            task_graph: TaskGraph,
            sorted_tasks: List[TaskTuple],
            task_encoder: Callable[[TaskAttr], List[float]]
    ) -> np.ndarray:
        """ Creates a list of task embeddings as per the MRLCO paper. """
        task_info = []
        for task_id, task_attr in sorted_tasks:
            task_predecessors = list(task_graph.pred[task_id].keys())
            task_successors = list(task_graph.succ[task_id].keys())
            encoded_task = task_encoder(task_attr)
            task_info.append(
                encoded_task
                + arrays.pad_list(lst=task_predecessors, target_length=TASK_PREDECESSORS, pad_value=-1.0)
                + arrays.pad_list(lst=task_successors, target_length=TASK_SUCCESSORS, pad_value=-1.0)
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
        embeddings[:, TASK_ID_COLUMNS] = normalize_task_ids(embeddings[:, TASK_ID_COLUMNS])
        embeddings[:, TASK_TIME_COLUMNS] = normalize_times(embeddings[:, TASK_TIME_COLUMNS])

        return embeddings
