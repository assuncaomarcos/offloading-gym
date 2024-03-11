#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gymnasium as gym
from .base import BaseOffEnv
from ..scheduler import Scheduler
from typing import Union, Optional, Tuple, List, Any, NamedTuple, Callable
from ..workload import Workload
from .workload import build_workload, RANDOM_WORKLOAD_CONFIG
from .scheduler import build_scheduler, DEFAULT_SCHEDULER_CONFIG
from ..task_graph import TaskGraph, TaskAttr
from ..cluster import Cluster
import jax.numpy as jnp
import numpy as np
import jax

# Number of downstream/upstream tasks considered when encoding a task graph
SUCCESSOR_TASKS = PREDECESSOR_TASKS = 6
TASK_PROFILE_LENGTH = 5


class OffloadingTask(TaskAttr):
    estimated_runtime: float
    rank: float

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.estimated_runtime = -1
        self.rank = -1


class OffloadingState(NamedTuple):
    task_embeddings: np.ndarray
    task_dependencies: np.ndarray
    scheduling_plan: np.ndarray


class TaskProfileEncoder(Callable):
    cluster: Cluster

    def __init__(self, cluster: Cluster):
        self.cluster = cluster

    def __call__(self, task_index: int, task: TaskAttr) -> List[Any]:
        local_exec_cost = self.cluster.local_execution_time(task.processing_demand)
        upload_cost = self.cluster.transmission_time(task.task_size)
        edge_exec_cost = self.cluster.edge_execution_time(task.processing_demand)
        download_cost = self.cluster.transmission_time(task.output_datasize)

        return [task_index, local_exec_cost, upload_cost, edge_exec_cost, download_cost]


class OffloadingEnv(BaseOffEnv):
    use_raw_state: bool
    observation_space: Union[gym.spaces.tuple.Tuple, gym.spaces.box.Box]
    action_space: gym.spaces.MultiBinary
    scheduler: Scheduler
    workload: Workload
    task_graph_space: Optional[gym.spaces.Box]
    task_dependency_space: Optional[gym.spaces.Box]
    scheduling_plan_space: Optional[gym.spaces.MultiBinary]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_raw_state = kwargs.get("use_raw_state", True)
        self._setup_spaces()

        self.workload_config = kwargs.get('workload', RANDOM_WORKLOAD_CONFIG)
        self.workload = build_workload(self.workload_config)
        self.scheduler_config = kwargs.get('scheduler', DEFAULT_SCHEDULER_CONFIG)
        self.scheduler = build_scheduler(self.scheduler_config)

    def _setup_spaces(self):
        self.action_space = gym.spaces.MultiBinary(self.tasks_per_app)
        self.setup_raw_state() if self.use_raw_state else self.setup_image_state()

    def setup_raw_state(self):
        self.task_graph_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(
                self.tasks_per_app,
                TASK_PROFILE_LENGTH
            )
        )

        self.task_dependency_space = gym.spaces.Box(
            low=0, high=1.0,
            shape=(
                self.tasks_per_app,
                SUCCESSOR_TASKS + PREDECESSOR_TASKS
            ),
        )

        self.scheduling_plan_space = gym.spaces.MultiBinary(self.tasks_per_app)

        self.observation_space = gym.spaces.Tuple(
            (self.task_graph_space, self.task_dependency_space, self.scheduling_plan_space)
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
        task_graphs = self.workload.step(offset=0)
        self.compute_task_ranks(task_graphs[0])

        # return self.state, {}

    def step(self, action: int) -> Tuple[OffloadingState, dict[str, Any]]:
        task_graphs = self.workload.step(offset=1)


    # def _get_obs(self):
    #     return {"agent": self._agent_location, "target": self._target_location}

    @property
    def state(self):
        pass
        # return OffloadingState(
        #     task_graph=self.tasks_per_app, TASK_PROFILE_LENGTH))
        # )

    def compute_task_ranks(self, task_graph: TaskGraph) -> List[OffloadingTask]:
        """ Computes the task ranks as per the MRLCO paper """
        tasks = []
        successors = task_graph.succ
        cluster = self.scheduler.cluster

        def task_runtime(task: OffloadingTask) -> float:
            if task.estimated_runtime > -1:
                return task.estimated_runtime

            local_exec = cluster.local_execution_time(task.processing_demand)
            upload = cluster.transmission_time(task.task_size)
            edge_exec = cluster.edge_execution_time(task.processing_demand)
            download = cluster.transmission_time(task.output_datasize)
            return min(local_exec, upload + edge_exec + download)

        def rank(task: OffloadingTask) -> float:
            if task.rank > -1:
                return task.rank

            task_successors = successors[task.task_id]
            if len(task_successors) == 0:
                task.rank = task_runtime(task)
            else:
                task.rank = task_runtime(task) + max(rank(task_graph.nodes[j]) for j in successors.keys())
                return task.rank

        for task_attr in task_graph.nodes().values():
            offloading_task = OffloadingTask(**task_attr)
            offloading_task.rank = rank(offloading_task)
            tasks.append(offloading_task)

        return tasks

#
#
# def task_graph_embedding(task_graph: TaskGraph) -> np.ndarray:
#     successors = task_graph.succ
#     predecessors = task_graph.pred




        # embedding = np.zeros(shape=self.task_graph_space.shape, dtype=np.float32)
        #
        # return embedding

    # def encode_graph(self, task_encoder: ABCTaskEncoding = None, task_sorting: ABCTaskSorting = None):
    #     sequence = []
    #     for task_idx in range(self.number_of_tasks):
    #         task = self.tasks[task_idx]
    #         predecessors, successors = [], []
    #
    #         for pred_idx in range(0, task_idx):
    #             if self.task_dependencies[pred_idx][task_idx] > 0.1:
    #                 predecessors.append(pred_idx)
    #
    #         for suc_idx in range(task_idx + 1, self.number_of_tasks):
    #             if self.task_dependencies[task_idx][suc_idx] > 0.1:
    #                 successors.append(suc_idx)
    #
    #         predecessors = pad_list(predecessors, TaskGraph.ENCODING_LENGTH)[0:TaskGraph.ENCODING_LENGTH]
    #         successors = pad_list(successors, TaskGraph.ENCODING_LENGTH)[0:TaskGraph.ENCODING_LENGTH]
    #
    #         if task_encoder is not None:
    #             encoded_task = task_encoder.encode_task(task_idx, task)
    #         else:
    #             encoded_task = [
    #                 self.normalize_datasize(task.proc_datasize),
    #                 self.normalize_datasize(task.trans_datasize)
    #             ]
    #
    #         sequence.append(encoded_task + predecessors + successors)
    #
    #     if task_sorting is not None:
    #         sequence = [sequence[i] for i in task_sorting.index_array(self.tasks, self.successors)]
    #
    #     return sequence

    # def index_array(self, tasks: List[Task], successors: Dict[int, List[int]]) -> np.ndarray:
    #     task_number = len(tasks)
    #     runtimes = self._compute_task_runtimes(tasks)
    #     rank_dict = [-1.0] * task_number
    #
    #     def rank(task_index):
    #         if rank_dict[task_index] != -1:
    #             return rank_dict[task_index]
    #
    #         if len(successors[task_index]) == 0:
    #             rank_dict[task_index] = runtimes[task_index]
    #             return rank_dict[task_index]
    #         else:
    #             rank_dict[task_index] = runtimes[task_index] + max(rank(j) for j in successors[task_index])
    #             return rank_dict[task_index]
    #
    #     for task_idx in range(task_number):
    #         rank(task_idx)
    #
    #     return np.argsort(rank_dict)[::-1]