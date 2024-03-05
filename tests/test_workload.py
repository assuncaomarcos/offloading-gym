#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests the random workload generator """

import unittest
from offloading_gym.envs.workload import RandomDAGParameters, DEFAULT_WORKLOAD_CONFIG, RandomDAGWorkload
import random

NUM_TASKS = 20


class TestWorkload(unittest.TestCase):
    """Tests the workload generator."""

    def setUp(self) -> None:
        random.seed(42)
        self.workload_params = RandomDAGParameters(**DEFAULT_WORKLOAD_CONFIG)
        self.workload = RandomDAGWorkload(tasks_per_app=NUM_TASKS, dag_parameters=self.workload_params)

    def tearDown(self) -> None:
        pass

    def test_create_taskgraph(self):
        task_graph = self.workload.step(1)[0]
        self.assertEqual(len(task_graph.nodes), NUM_TASKS)


if __name__ == "__main__":
    unittest.main()
