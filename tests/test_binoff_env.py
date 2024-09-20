import unittest
import networkx as nx
import gymnasium as gym
import numpy as np
import re
from offloading_gym.simulation.offload import Cluster, Simulator
from offloading_gym.task_graph import TaskAttr, EdgeAttr, TaskGraph

CPU_UNIT = 10 ** 9  # It takes one second to execute a task

# To ensure it takes one second to transfer it over the channel
DATA_UNIT = int(10 ** 6 / 8)

# Set all power consumption to 1.25W
POWER_UNIT = 1.25

TEST_CLUSTER_CONFIG = {
    "num_edge_cpus": 4,
    "edge_cpu_capacity": 4 * 10 ** 9,
    "num_local_cpus": 1,
    "local_cpu_capacity": 10 ** 9,
    "upload_rate": 1,
    "download_rate": 1,
    "power_tx": 1.2,
    "power_rx": 1.2,
    "power_cpu": 1.2,
}

TEST_WORKLOAD_CONFIG = {
    "type": "random_dag",
    "min_computing": 10 ** 7,
    "max_computing": 10 ** 8,
    "min_datasize": 5120,
    "max_datasize": 51200,
    "density_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "regularity_values": [0.2, 0.5, 0.8],
    "fat_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "ccr_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "jump_values": [1, 2, 4],
}

RNG_SEED = 42


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


class TestOffloadingEnv(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(RNG_SEED)
        np.set_printoptions(suppress=True)

    def test_env_instantiation(self):
        try:
            gym.make(
                "BinaryOffload-v0",
                **{"tasks_per_app": 30},
            )
        except gym.error.Error as error:
            self.fail(f"Unexpected error: {error}")

    def test_cluster_config(self):
        try:
            env = gym.make(
                "BinaryOffload-v0",
                **{"tasks_per_app": 30, "cluster": TEST_CLUSTER_CONFIG},
            )
            cluster = env.get_wrapper_attr("cluster")
            self.assertEqual(cluster.num_edge_cpus, 4)
            self.assertEqual(cluster.power_rx, 1.2)
        except gym.error.Error as error:
            self.fail(f"Unexpected error: {error}")

    def test_workload_config(self):
        try:
            env = gym.make(
                "BinaryOffload-v0",
                **{"tasks_per_app": 30, "workload": TEST_WORKLOAD_CONFIG},
            )
            workload = env.get_wrapper_attr("workload")
            self.assertEqual(workload.num_tasks, [30])
        except gym.error.Error as error:
            self.fail(f"Unexpected error: {error}")

    def test_no_normalized_ids(self):
        num_tasks = 20
        env = gym.make(
            "BinaryOffload-v0",
            **{"tasks_per_app": num_tasks, "normalize_task_ids": False},
        )
        env.action_space.seed(seed=RNG_SEED)
        self.assertTrue(np.all(env.observation_space.high == num_tasks))

        _ = env.reset(seed=RNG_SEED)
        action = env.action_space.sample()
        env.step(action)

    def test_env_reset(self):
        env = gym.make(
            "BinaryOffload-v0",
            **{"tasks_per_app": 20},
        )
        env.action_space.seed(seed=RNG_SEED + 1)
        _ = env.reset(seed=RNG_SEED + 1)
        action = env.action_space.sample()
        env.step(action)

    def test_multiple_resets(self):
        env = gym.make(
            "BinaryOffload-v0",
            **{"tasks_per_app": 30},
        )
        for i in range(20):
            _ = env.reset(seed=RNG_SEED + i)

    def test_vector_env(self):
        num_envs = 3
        num_tasks = 30
        envs = gym.make_vec(
            "BinaryOffload-v0",
            num_envs=num_envs,
            **{"tasks_per_app": num_tasks},
        )
        self.assertEqual(envs.action_space.shape[0], num_envs)
        self.assertEqual(envs.action_space.shape[1], num_tasks)

        for i in range(5):
            obs_list, _ = envs.reset(seed=RNG_SEED + i)
            self.assertEqual(len(obs_list), num_envs)
