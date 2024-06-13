import unittest
import networkx as nx
from ..simulation.offload import Cluster, Simulator
from ..task_graph import TaskAttr, EdgeAttr, TaskGraph

CPU_UNIT = 10**9  # It takes one second to execute a task

# To ensure it takes one second to transfer it over the channel
DATA_UNIT = int(10**6 / 8)

# Set all power consumption to 1.25W
POWER_UNIT = 1.25


class TestSimulation(unittest.TestCase):

    def setUp(self) -> None:
        self.cluster = Cluster(
            num_edge_cpus=1,
            edge_cpu_capacity=CPU_UNIT,
            num_local_cpus=1,
            local_cpu_capacity=CPU_UNIT,
            upload_rate=1,
            download_rate=1,
            power_rx=POWER_UNIT,
            power_tx=POWER_UNIT,
            power_cpu=POWER_UNIT,
        )

    def test_build(self):
        self.assertEqual(type(Simulator.build(self.cluster)), Simulator)

    def test_pipeline(self):
        task_graph = self._generate_pipeline()
        topo_order = nx.topological_sort(task_graph)
        sorted_tasks = [(node, task_graph.nodes[node]) for node in topo_order]
        task_info = Simulator.build(self.cluster).simulate(sorted_tasks, [1, 0, 1, 0])
        self.assertEqual(len(task_info), 4)
        finish_times = [3.0, 4.0, 7.0, 8.0]
        for idx, ft in zip(range(4), finish_times):
            self.assertEqual(task_info[idx].finish_time, ft)

    def test_diamond_dag(self):
        task_graph = self._generate_diamond_dag()
        topo_order = nx.topological_sort(task_graph)
        sorted_tasks = [(node, task_graph.nodes[node]) for node in topo_order]
        task_info = Simulator.build(self.cluster).simulate(sorted_tasks, [1, 0, 1, 0])
        finish_times = [3.0, 4.0, 7.0, 8.0]
        for idx, ft in zip(range(4), finish_times):
            self.assertEqual(task_info[idx].finish_time, ft)

    def test_multiple_steps(self):
        task_graph = self._generate_diamond_dag()
        topo_order = nx.topological_sort(task_graph)
        sorted_tasks = [(node, task_graph.nodes[node]) for node in topo_order]
        sim = Simulator.build(self.cluster)
        task_info = sim.simulate(sorted_tasks[:2], [1, 0])
        self.assertEqual(task_info[0].finish_time, 3.0)
        self.assertEqual(task_info[1].finish_time, 4.0)
        task_info = sim.simulate(sorted_tasks[2:], [1, 0])
        finish_times = [3.0, 4.0, 7.0, 8.0]
        for idx, ft in zip(range(4), finish_times):
            self.assertEqual(task_info[idx].finish_time, ft)

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
            (2, 3, EdgeAttr(datasize=DATA_UNIT)),
        ]

        task_graph = TaskGraph()
        task_graph.add_nodes_from(tasks)
        task_graph.add_edges_from(edges)
        return task_graph
