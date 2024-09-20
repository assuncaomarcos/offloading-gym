#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests the random workload generator """

import unittest
from offloading_gym.workload import RandomDAGGenerator
from offloading_gym.workload import daggen
import numpy as np

WORKLOAD_CONFIG = {
    "num_tasks": [10, 15, 20, 50],  # Make sure this is set when using this config
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

RNG_SEED = 42


class TestDaggen(unittest.TestCase):
    """Tests the DAG generator."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(RNG_SEED)

    def test_daggen(self):
        """Tests generating graph with daggen"""
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
            dags.append(
                daggen.random_dag(
                    rng=self.rng, num_tasks=n, density=v, fat=v, ccr=0.8, jump=2
                )
            )

        for dag, num_task in zip(dags, num_tasks):
            self.assertEqual(len(dag.nodes), num_task)


class TestWorkload(unittest.TestCase):
    """Tests the workload generator."""

    def setUp(self) -> None:
        self.num_tasks = WORKLOAD_CONFIG["num_tasks"]
        self.workload = RandomDAGGenerator(**WORKLOAD_CONFIG)

    def test_create_taskgraph(self):
        """Test generating a couple task graphs"""
        self.workload.reset(seed=RNG_SEED)
        for _ in range(5):
            task_graph = self.workload.step(offset=1)[0]
            self.assertIn(
                len(task_graph.tasks),
                self.num_tasks,
                "Graph does not have a size specified in the config",
            )

    def test_data_sizes(self):
        self.workload.reset(seed=RNG_SEED)
        for _ in range(2):
            task_graph = self.workload.step(offset=1)[0]
            for task in task_graph.tasks.values():
                self.assertTrue(
                    WORKLOAD_CONFIG["min_datasize"]
                    <= task.task_size
                    <= WORKLOAD_CONFIG["max_datasize"]
                )


if __name__ == "__main__":
    unittest.main()
