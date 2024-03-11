#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests the random workload generator """

import unittest
from offloading_gym.workload import RandomDAGGenerator
from offloading_gym.envs.workload import RANDOM_WORKLOAD_CONFIG
from ..workload import daggen
import random


class TestDaggen(unittest.TestCase):
    """Tests the DAG generator."""

    def setUp(self) -> None:
        random.seed(42)

    def test_daggen(self):
        task_graph = daggen.random_dag(num_tasks=20)
        pattern = r"DiGraph with \d+ nodes and \d+ edges"
        self.assertRegex(str(task_graph), pattern)

    def test_random_graphs(self):
        # Used to vary the values of fat and density parameters
        values = [0.4, 0.6, 0.8, 1.0]

        # Number of tasks in each generated DAG
        num_tasks = [7, 10, 15, 15]

        dags = []  # to store the DAGs

        for v, n in zip(values, num_tasks):
            dags.append(daggen.random_dag(num_tasks=n, density=v, fat=v, ccr=0.8, jump=2))

        for dag, num_task in zip(dags, num_tasks):
            self.assertEqual(len(dag.nodes), num_task)


class TestWorkload(unittest.TestCase):
    """Tests the workload generator."""

    def setUp(self) -> None:
        random.seed(42)
        self.num_tasks = RANDOM_WORKLOAD_CONFIG["num_tasks"]
        self.workload = RandomDAGGenerator(**RANDOM_WORKLOAD_CONFIG)

    def test_create_taskgraph(self):
        task_graph = self.workload.step(1)[0]
        self.assertEqual(len(task_graph.nodes), self.num_tasks)

    def test_successors(self):
        task_graph = self.workload.step(1)[0]
        successors = task_graph.succ
        self.assertEqual(successors[1][2]['datasize'], 809)


if __name__ == "__main__":
    unittest.main()
