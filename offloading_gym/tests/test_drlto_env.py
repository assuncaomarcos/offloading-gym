import unittest
import gymnasium as gym
import numpy as np


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

    def test_no_normalized_ids(self):
        env = gym.make(
            "BinaryOffload-v0",
            **{"tasks_per_app": 20, "normalized_task_ids": False},
        )
        env.action_space.seed(seed=5)
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
        envs = gym.make_vec(
            "BinaryOffload-v0",
            num_envs=num_envs,
            **{"tasks_per_app": 30},
        )
        self.assertEqual(envs.action_space.shape[0], num_envs)
        self.assertEqual(envs.action_space.shape[1], 30)

        for i in range(5):
            obs_list, _ = envs.reset(seed=5)
            self.assertEqual(len(obs_list), num_envs)
