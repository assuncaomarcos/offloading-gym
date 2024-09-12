#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides mixins for offloading gym environments.
"""

from typing import Callable

from offloading_gym.task_graph import TaskGraph, TaskAttr


class TaskGraphMixin:

    @staticmethod
    def compute_task_ranks(
        task_graph: TaskGraph, task_runtime_fn: Callable[[TaskAttr], float]
    ):
        """Computes the task ranks as per the MRLCO/DRLTO papers."""
        successors = task_graph.succ

        def task_rank(task: TaskAttr) -> float:
            rank = task.get("rank")
            if rank is not None:
                return rank

            runtime = task["estimated_runtime"]
            task_successors = successors[task.task_id]
            task["rank"] = runtime

            if task_successors:
                task["rank"] = runtime + max(
                    task_rank(task_graph.nodes[j]) for j in task_successors.keys()
                )
            return runtime

        for task_attr in task_graph.nodes().values():
            task_attr["estimated_runtime"] = task_runtime_fn(task_attr)

        for task_attr in task_graph.nodes().values():
            task_attr["rank"] = task_rank(task_attr)
