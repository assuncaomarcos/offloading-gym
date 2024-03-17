import unittest
import networkx as nx
from ..simulator import Cluster, Simulator
from ..task_graph import TaskAttr, EdgeAttr, TaskGraph

CPU_UNIT = 1024 * 1024          # It takes one second to execute a task
DATA_UNIT = int(10 ** 6 / 8)    # To ensure it takes one second to transfer it over the channel


class TestSimulation(unittest.TestCase):

    def setUp(self) -> None:
        self.cluster = Cluster(
            num_edge_cpus=1,
            edge_cpu_capacity=CPU_UNIT,
            num_local_cpus=1,
            local_cpu_capacity=CPU_UNIT,
            upload_rate=1,
            download_rate=1
        )

    def test_build(self):
        self.assertEqual(type(Simulator.build(self.cluster)), Simulator)

    def test_pipeline(self):
        task_graph = self._generate_pipeline()
        topo_order = nx.topological_sort(task_graph)
        sorted_tasks = [(node, task_graph.nodes[node]) for node in topo_order]
        task_info = Simulator.build(self.cluster).simulate(sorted_tasks, [1, 0, 1, 0])
        self.assertEqual(len(task_info), 4)
        self.assertEqual(task_info[0].finish_time, 3.0)
        self.assertEqual(task_info[1].finish_time, 4.0)
        self.assertEqual(task_info[2].finish_time, 7.0)
        self.assertEqual(task_info[3].finish_time, 8.0)

    def test_diamond_dag(self):
        task_graph = self._generate_diamond_dag()
        topo_order = nx.topological_sort(task_graph)
        sorted_tasks = [(node, task_graph.nodes[node]) for node in topo_order]
        task_info = Simulator.build(self.cluster).simulate(sorted_tasks, [1, 0, 1, 0])
        self.assertEqual(task_info[0].finish_time, 3.0)
        self.assertEqual(task_info[1].finish_time, 4.0)
        self.assertEqual(task_info[2].finish_time, 7.0)
        self.assertEqual(task_info[3].finish_time, 8.0)

    @staticmethod
    def _generate_pipeline() -> TaskGraph:
        tasks, edges = [], []
        for task_id in range(4):
            tasks.append(
                (
                    task_id,
                    TaskAttr(
                        task_id=task_id,
                        processing_demand=CPU_UNIT,
                        task_size=DATA_UNIT,
                        output_datasize=DATA_UNIT,
                    ),
                )
            )
        for src in range(3):
            edges.append(
                (
                    src,
                    src + 1,
                    EdgeAttr(datasize=DATA_UNIT),
                )
            )
        task_graph = TaskGraph()
        task_graph.add_nodes_from(tasks)
        task_graph.add_edges_from(edges)
        return task_graph

    @staticmethod
    def _generate_diamond_dag() -> TaskGraph:
        tasks, edges = [], []
        for task_id in range(4):
            tasks.append(
                (
                    task_id,
                    TaskAttr(
                        task_id=task_id,
                        processing_demand=CPU_UNIT,
                        task_size=DATA_UNIT,
                        output_datasize=DATA_UNIT,
                    ),
                )
            )

        edges = [
            (0, 1, EdgeAttr(datasize=DATA_UNIT)),
            (0, 2, EdgeAttr(datasize=DATA_UNIT)),
            (1, 3, EdgeAttr(datasize=DATA_UNIT)),
            (2, 3, EdgeAttr(datasize=DATA_UNIT))
        ]

        task_graph = TaskGraph()
        task_graph.add_nodes_from(tasks)
        task_graph.add_edges_from(edges)
        return task_graph
