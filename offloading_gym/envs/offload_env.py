#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides a Gymnasium environment, `BinaryEnv`, designed for training
Deep Reinforcement Learning (DRL) algorithms that will offload tasks from local
devices to edge servers.

The environment employs a workload generator to facilitate creating sample applications.
These applications are structured as Directed Acyclic Graphs (DAGs), where nodes represent
tasks and edges symbolize data interdependencies.

The implementation of this environment aligns with the descriptions in the
following research papers:

- Wang, Jin et al. "Dependent task offloading for edge computing based on deep
  reinforcement learning." IEEE Transactions on Computers 71, no. 10 (2021): 2449-2461.

- Wang, Jin et al. "Fast adaptive task offloading in edge computing based on
  meta reinforcement learning." IEEE Transactions on Parallel and Distributed Systems
  32, no. 1 (2020): 242-253.

Contrary to the above research, this workload generator dynamically creates task
graphs whose structure follow the patterns used in the papers above. The workload
generator furthermore allows for customization to suit various users' needs.

The workload generation is based on the daggen random graph generator
suggested by Suter & Hunold with modifications from the following paper:

- H. Arabnejad and J. Barbosa. "List Scheduling Algorithm for Heterogeneous Systems by
  an Optimistic Cost Table." IEEE Transactions on Parallel and Distributed Systems,
  Vol. 25, No. 3, March 2014.

The `BinaryEnv` environment mimics a simple computing infrastructure:

       +-------------+      Upload       +------------+
       |             | ----------------> |            |
       | User Device |                   | Edge Server|
       |             | <---------------- |            |
       +-------------+      Download     +------------+

This infrastructure includes a user device connected to an edge server.
Tasks can be offloaded from the device to the server. These resources are interlinked
by a network connection with distinct capacities for upload (device to edge server)
and download (edge server to device).
"""

from typing import Union, Optional, Tuple, List, Any, Callable, Dict
from numpy.typing import NDArray
from functools import cached_property

import gymnasium as gym
import numpy as np
import networkx as nx
import math

from offloading_gym.envs.base import BaseOffEnv
from offloading_gym.envs.workload import RandomGraphWorkload
from offloading_gym.task_graph import TaskGraph, TaskAttr, TaskTuple
from offloading_gym.utils import arrays
from offloading_gym.workload import Workload
from offloading_gym.simulation.offload import Cluster, Simulator, TaskExecution


__all__ = ["BinaryOffloadEnv"]

TASKS_PER_APPLICATION = 20

# Number of embedding fields containing task properties
TASK_PROFILE_LENGTH = 5

# Number of downstream/upstream tasks considered when encoding a task graph
TASK_SUCCESSORS = TASK_PREDECESSORS = 6

# Columns of the graph embedding that contain task tims and IDs
TASK_TIME_COLUMNS = list(range(1, TASK_PROFILE_LENGTH))
TASK_ID_COLUMNS = [0] + list(
    range(
        TASK_PROFILE_LENGTH, TASK_PROFILE_LENGTH + TASK_PREDECESSORS + TASK_SUCCESSORS
    )
)

DEFAULT_CLUSTER_CONFIG = {
    "num_edge_cpus": 1,
    "edge_cpu_capacity": 4 * 10**9,
    "num_local_cpus": 1,
    "local_cpu_capacity": 10**9,
    "upload_rate": 11,
    "download_rate": 11,
    "power_tx": 1.258,
    "power_rx": 1.181,
    "power_cpu": 1.25,
}

DEFAULT_WORKLOAD_CONFIG = {
    "num_tasks": [20],       # Make sure this is set when using this config
    "min_computing": 10**7,  # Each task requires between 10^7 and 10^8 cycles
    "max_computing": 10**8,
    "min_datasize": 5120,  # Each task produces between 5KB and 50KB of data
    "max_datasize": 51200,
    "density_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "regularity_values": [0.2, 0.5, 0.8],
    "fat_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "ccr_values": [0.3, 0.4, 0.5],
    "jump_values": [1, 2],
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
        return [
            float(task.task_id),
            local_exec_cost,
            upload_cost,
            edge_exec_cost,
            download_cost,
        ]


class BinaryOffloadEnv(BaseOffEnv):
    workload: Workload
    cluster: Cluster
    tasks_per_app: int

    observation_space: gym.spaces.Box
    action_space: gym.spaces.MultiBinary

    # Current task graph
    task_graph: Union[TaskGraph, None]

    # A numpy array containing the encoded task graph
    scheduling_plan: NDArray[np.int8]
    graph_embedding: Union[NDArray, None]

    # Tasks sorted for scheduling for avoiding having to sort them multiple times
    task_list: Union[List[TaskTuple], None]

    # To keep the results of an all-local execution to compute the reward
    local_execution: Union[List[TaskExecution], None]

    # The weights of latency and energy (lambda's) for computing the rewards
    weight_latency: float
    weight_energy: float

    # To truncate episodes
    max_episode_steps: Union[int, None]
    steps: int

    # To indicate whether task ids should be normalized
    normalize_task_ids: bool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks_per_app = kwargs.get("tasks_per_app", TASKS_PER_APPLICATION)
        self.max_episode_steps = kwargs.get("max_episode_steps", None)
        self.normalize_task_ids = kwargs.get("normalize_task_ids", True)

        self.weight_latency = kwargs.get("weight_latency", 0.5)
        self.weight_energy = kwargs.get("weight_energy", 0.5)
        assert math.isclose(
            self.weight_latency + self.weight_energy, 1.0
        ), "The sum of weight_latency and weight_energy must be 1.0"

        self._setup_spaces()
        self._build_simulation(kwargs)
        self.task_graph = None
        self.graph_embedding = None
        self.task_list = None
        self.local_execution = None
        self.steps = 0

    def _build_simulation(self, kwargs):
        workload_config = kwargs.get("workload", DEFAULT_WORKLOAD_CONFIG)
        workload_config["num_tasks"] = self.tasks_per_app
        self.workload = RandomGraphWorkload.build(workload_config)

        cluster_config = kwargs.get("cluster", DEFAULT_CLUSTER_CONFIG)
        self.cluster = Cluster(**cluster_config)
        self.task_encoder = TaskCostEncoder(self.cluster)

    def _setup_spaces(self):
        self.action_space = gym.spaces.MultiBinary(self.tasks_per_app)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0 if self.normalize_task_ids else self.tasks_per_app,
            shape=(
                self.tasks_per_app,
                TASK_PROFILE_LENGTH + TASK_SUCCESSORS + TASK_PREDECESSORS,
            ),
            dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.workload.reset(seed=seed)
        self.steps = 0

        # Use only the first task graph
        self.task_graph = self.workload.step(offset=0)[0]
        self._compute_task_ranks(self.cluster, self.task_graph)
        self.scheduling_plan = np.array(self.all_local_action, dtype=np.int8)

        topo_order = nx.lexicographical_topological_sort(
            self.task_graph, key=lambda task_id: self.task_graph.nodes[task_id]["rank"]
        )

        # Store sorted tasks to avoid having to sort it multiple times
        self.task_list = [(node, self.task_graph.nodes[node]) for node in topo_order]

        # Compute the execution times for a run where all tasks are executed locally
        self.local_execution = Simulator.build(self.cluster).simulate(
            self.task_list, self.all_local_action
        )

        self.graph_embedding = self._task_embeddings(
            task_graph=self.task_graph,
            sorted_tasks=self.task_list,
            task_encoder=self.task_encoder,
            normalize_ids=self.normalize_task_ids
        )

        return self._get_ob(), {}

    def _get_ob(self):
        s = self.state
        assert s is not None, "Call reset before using OffloadingEnv."
        return self.graph_embedding

    def step(self, action: NDArray[np.int8]) -> Tuple[
        NDArray[np.float32],
        np.float32,
        bool,
        bool,
        Dict[str, Any],
    ]:
        self.scheduling_plan = action
        self.steps += 1
        action_execution = Simulator.build(self.cluster).simulate(
            self.task_list, action.tolist()
        )
        action_make_span = np.array(
            [task_execution.make_span for task_execution in action_execution], dtype=np.float32
        )
        action_energy = np.array(
            [task_execution.energy for task_execution in action_execution], dtype=np.float32
        )

        scores_make_span = self._compute_scores(
            action_make_span, self.local_exec_make_span
        )
        scores_energy = self._compute_scores(action_energy, self.local_exec_energy)
        rewards = (
            self.weight_latency * scores_make_span + self.weight_energy * scores_energy
        )

        truncate = (
            self.max_episode_steps is not None and self.steps >= self.max_episode_steps
        )

        return (
            self._get_ob(),
            # np.sum(rewards, axis=0),
            np.mean(rewards),
            False,
            truncate,
            {
                "task_rewards": rewards,
                "task_energy": action_energy,
                "task_graph": self.task_graph
            },
        )

    def _compute_scores(
        self,
        action_results: NDArray[np.float32],
        local_results: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        avg_local = local_results / float(self.tasks_per_app)
        scores = -(action_results - avg_local) / local_results
        return scores

    @cached_property
    def all_local_action(self):
        return [0] * self.tasks_per_app

    @property
    def local_exec_make_span(self) -> NDArray[np.float32]:
        return np.array(
            [task_execution.make_span for task_execution in self.local_execution], dtype=np.float32
        )

    @property
    def local_exec_energy(self) -> NDArray[np.float32]:
        return np.array(
            [task_execution.energy for task_execution in self.local_execution], dtype=np.float32
        )

    @property
    def state(self) -> NDArray[np.float32]:
        return self.graph_embedding

    @staticmethod
    def _compute_task_ranks(cluster: Cluster, task_graph: TaskGraph):
        """Computes the task ranks as per the MRLCO/DRLTO papers."""
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
                task["rank"] = runtime + max(
                    task_rank(task_graph.nodes[j]) for j in task_successors.keys()
                )
                return task["rank"]

        for task_attr in task_graph.nodes().values():
            task_attr["estimated_runtime"] = task_runtime(task_attr)

        for task_attr in task_graph.nodes().values():
            task_attr["rank"] = task_rank(task_attr)

    @staticmethod
    def _task_embeddings(
        task_graph: TaskGraph,
        sorted_tasks: List[TaskTuple],
        task_encoder: Callable[[TaskAttr], List[float]],
        normalize_ids: bool = True
    ) -> np.ndarray:
        """Creates a list of task embeddings as per the MRLCO/DRLTO papers."""
        task_info = []
        for task_id, task_attr in sorted_tasks:
            task_predecessors = list(task_graph.pred[task_id].keys())
            task_predecessors = arrays.pad_list(
                lst=task_predecessors,
                target_length=TASK_PREDECESSORS,
                pad_value=-1.0,
            )

            task_successors = list(task_graph.succ[task_id].keys())
            task_successors = arrays.pad_list(
                lst=task_successors, target_length=TASK_SUCCESSORS, pad_value=-1.0
            )

            task_embedding = task_encoder(task_attr) + task_predecessors + task_successors
            task_info.append(task_embedding)

        def normalize_task_ids(dependencies: np.ndarray) -> np.ndarray:
            mask = dependencies != -1
            min_val = dependencies[mask].min()
            max_val = dependencies[mask].max()
            dependencies[mask] = (dependencies[mask] - min_val) / (max_val - min_val)
            return dependencies

        def normalize_times(features: np.ndarray) -> np.ndarray:
            return (features - features.min()) / (features.max() - features.min())

        embeddings = np.array(task_info, dtype=np.float32)

        if normalize_ids:
            embeddings[:, TASK_ID_COLUMNS] = normalize_task_ids(
                embeddings[:, TASK_ID_COLUMNS]
            )

        embeddings[:, TASK_TIME_COLUMNS] = normalize_times(
            embeddings[:, TASK_TIME_COLUMNS]
        )

        return embeddings
