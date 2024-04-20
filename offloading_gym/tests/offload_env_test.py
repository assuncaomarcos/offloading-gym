import unittest
import gymnasium as gym
import numpy as np

from offloading_gym.envs import BinaryOffloadWrapper


TEST_CLUSTER_CONFIG = {
    "num_edge_cpus": 4,
    "edge_cpu_capacity": 4 * 10**9,
    "num_local_cpus": 1,
    "local_cpu_capacity": 10**9,
    "upload_rate": 1,
    "download_rate": 1,
    "power_tx": 1.2,
    "power_rx": 1.2,
    "power_cpu": 1.2,
}


TEST_WORKLOAD_CONFIG = {
    "type": "random_dag",
    "min_computing": 10**7,
    "max_computing": 10**8,
    "min_datasize": 5120,
    "max_datasize": 51200,
    "density_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "regularity_values": [0.2, 0.5, 0.8],
    "fat_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "ccr_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "jump_values": [1, 2, 4],
}


class TestOffloadingEnv(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
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
            self.assertEqual(workload.num_tasks, 30)
        except gym.error.Error as error:
            self.fail(f"Unexpected error: {error}")

    def test_no_normalized_ids(self):
        num_tasks = 20
        env = gym.make(
            "BinaryOffload-v0",
            **{"tasks_per_app": num_tasks, "normalize_task_ids": False},
        )
        env.action_space.seed(seed=5)
        self.assertTrue(np.all(env.observation_space.high == num_tasks))
        _ = env.reset(seed=5)
        action = env.action_space.sample()
        env.step(action)

    def test_env_reset(self):
        env = gym.make(
            "BinaryOffload-v0",
            **{"tasks_per_app": 20},
        )
        env.action_space.seed(seed=5)
        _ = env.reset(seed=5)
        action = env.action_space.sample()
        env.step(action)

    def test_multiple_resets(self):
        env = gym.make(
            "BinaryOffload-v0",
            **{"tasks_per_app": 30},
        )
        for i in range(20):
            _ = env.reset(seed=5)

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
            obs_list, _ = envs.reset(seed=5)
            self.assertEqual(len(obs_list), num_envs)

    def test_wrapper(self):
        num_tasks = 30
        env = gym.make(
            "BinaryOffload-v0",
            **{"tasks_per_app": num_tasks, "max_episode_steps": 1},
        )
        wrapped_env = BinaryOffloadWrapper(gym_env=env)
        self.assertEqual(wrapped_env.action_spec().shape, (30, ))
        self.assertEqual(wrapped_env.observation_spec().shape, (30, 17))
        self.assertEqual(wrapped_env.observation_spec().minimum, -1.0)
        wrapped_env.reset()
        ts = wrapped_env.step(action=env.action_space.sample())
        self.assertTrue(isinstance(ts.reward, np.ndarray))
