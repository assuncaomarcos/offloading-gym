#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Tests the task parser and task graphs """

import unittest
from offloading_gym.workload import daggen
import networkx as nx
import random


class TestDaggen(unittest.TestCase):
    """Tests the task generator."""

    def setUp(self) -> None:
        random.seed(42)

    def tearDown(self) -> None:
        pass

    def test_daggen(self):
        task_graph = daggen.daggen()
        print(task_graph)
        for task in task_graph.nodes.values():
            print(task)

        for edge in task_graph.edges:
            print(edge)

    def test_dot_file(self):
        task_graph = daggen.daggen()
        nx.drawing.nx_agraph.write_dot(task_graph, "./dag.dot")

    # def test_plot(self):
    #     import networkx as nx
    #     import matplotlib.pyplot as plt
    #
    #     # To seed daggen
    #     seed = 46464
    #
    #     # Used to vary the values of fat and density parameters
    #     values = [0.4, 0.6, 0.8, 1.0]
    #
    #     # Number of tasks in each generated DAG
    #     num_tasks = [7, 10, 15, 15]
    #
    #     dags = []  # to store the DAGs
    #
    #     for v, n in zip(values, num_tasks):
    #         dags.append(daggen.daggen(num_tasks=n, density=v, fat=v, ccr=0.8, jump=2))
    #
    #     height = 10  # height of a plot
    #     width = 5  # width of a plot
    #
    #     fig, axs = plt.subplots(1, len(values), figsize=(height, width))
    #
    #     for i, dag in enumerate(dags):
    #         for layer, nodes in enumerate(sorted(nx.topological_generations(dag), reverse=True)):
    #             # multipartite_layout expects the layer as a node attribute,
    #             # so add the numeric layer value as a node attribute
    #             for node in nodes:
    #                 dag.nodes[node]["layer"] = layer
    #
    #         # Compute the multipartite_layout using the "layer" node attribute
    #         pos = nx.multipartite_layout(dag, subset_key="layer", align='horizontal')
    #         nx.draw(dag, pos=pos, ax=axs[i], with_labels=True,
    #                 node_color="lightblue", node_size=400, font_family="sans-serif")
    #
    #     # Draw arrow
    #     plt.annotate('', xy=(0.20, 0.10), xycoords='figure fraction', xytext=(0.8, 0.10),
    #                  arrowprops=dict(arrowstyle="<->", color='black'))
    #     plt.text(0.25, 0.05, "Low fat and density", fontsize=12, transform=plt.gcf().transFigure)
    #     plt.text(0.6, 0.05, "High fat and density", fontsize=12, transform=plt.gcf().transFigure)
    #     plt.savefig('daggen.pdf')


if __name__ == "__main__":
    unittest.main()
