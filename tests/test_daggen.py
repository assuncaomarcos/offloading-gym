#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests the task parser and task graphs """

import unittest
from offloading_gym.workload import daggen
import random


class TestDaggen(unittest.TestCase):
    """Tests the task generator."""

    def setUp(self) -> None:
        random.seed(42)

    def tearDown(self) -> None:
        pass

    def test_daggen(self):
        task_graph = daggen.daggen()
        self.assertEqual(str(task_graph), "DiGraph with 20 nodes and 25 edges")

    def test_random_graphs(self):
        # Used to vary the values of fat and density parameters
        values = [0.4, 0.6, 0.8, 1.0]

        # Number of tasks in each generated DAG
        num_tasks = [7, 10, 15, 15]

        dags = []  # to store the DAGs

        for v, n in zip(values, num_tasks):
            dags.append(daggen.daggen(num_tasks=n, density=v, fat=v, ccr=0.8, jump=2))

        for dag, num_task in zip(dags, num_tasks):
            self.assertEqual(len(dag.nodes), num_task)


if __name__ == "__main__":
    unittest.main()
