#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests the task parser and task graphs """

import unittest
from offloading_gym.task_graph import TaskGraph
import random


class TestTaskParser(unittest.TestCase):
    """Tests the task parser."""

    def setUp(self) -> None:
        random.seed(42)

    def tearDown(self) -> None:
        pass

    # def test_taskgraph_creation(self):
    #     g = TaskGraph()
    #     g.add_nodes_from([1, 2])
    #     for t in g.tasks.values():
    #         print(t)

    # def test_graph_parser(self):
    #     task_graph = parse_dot("tests/samples/task_graph.gv")
    #     number_tasks = len(task_graph.tasks)
    #     self.assertEqual(number_tasks, 20)
        # for task in task_graph.tasks.values():
        #     print(task)
        #
        # for edge in task_graph.edges.values():
        #     print(edge)

        # self.assertEqual(task_graph.max_datasize, 170280960)
        # self.assertEqual(task_graph.min_datasize, 5270528)
        # self.assertEqual(task_graph.tasks[0].proc_datasize, 162501973)
        # self.assertEqual(task_graph.tasks[0].trans_datasize, 48750592)
        # self.assertEqual(task_graph.tasks[14].depth, 3)

    # def test_daggen_graph(self):
    #     daggen_graph(num_tasks=10)
    #
    # @staticmethod
    # def _normalize_datasize(datasize, task_graph):
    #     return float(datasize - task_graph.min_datasize) / float(task_graph.max_datasize - task_graph.min_datasize)
    #
    # def test_graph_encoding(self):
    #     task_graph = TaskGraphParser.parse_graph("tests/task_graph.gv")
    #     first_task = task_graph.tasks[0]
    #     encoding = task_graph.encode_graph()
    #     first_encoding = encoding[0]
    #     self.assertEqual(first_encoding[0], self._normalize_datasize(first_task.proc_datasize, task_graph))
    #     last_encoding = encoding[-1]
    #     last_task = task_graph.tasks[-1]
    #     self.assertEqual(last_encoding[1], self._normalize_datasize(last_task.trans_datasize, task_graph))


if __name__ == "__main__":
    unittest.main()
