import unittest
import gymnasium as gym
import numpy as np


class TestFogEnv(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        np.set_printoptions(suppress=True)

    def test_env_instantiation(self):
        try:
            gym.make(
                "FogPlacement-v0",
                **{},
            )
        except gym.error.Error as error:
            self.fail(f"Unexpected error: {error}")

    def test_reset_env(self):
        env = gym.make(
            "FogPlacement-v0",
            **{},
        )
        env.reset(seed=42)

    def test_execute_on_iot(self):
        env = gym.make(
            "FogPlacement-v0",
            **{},
        )
        env.reset(seed=42)
        env.step(action=0)
        env.step(action=0)

    def test_random_execution(self):
        env = gym.make(
            "FogPlacement-v0",
            **{},
        )
        env.action_space.seed(42)
        env.reset(seed=42)

        for _ in range(50):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()
