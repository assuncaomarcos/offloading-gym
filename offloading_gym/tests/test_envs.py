import unittest
import gymnasium as gym
import numpy as np
from ..envs.offloading import OffloadingEnv


class TestOffloadingEnv(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

    def test_instantiation_with_gym(self):
        gym.make('Offloading-v0')

    def test_environment_reset(self):
        env: OffloadingEnv = gym.make(  # type: ignore
            "Offloading-v0",
            **{
                'tasks_per_app': 20
            },
        )
        env.action_space.seed(seed=5)
        _ = env.reset(seed=5)
        action = env.action_space.sample()
        env.step(action)

    #     # action = 0
    #     # done = False
    #     # while not done:
    #     #     _, _, done, _ = env.step(action)
    #     # self.assertTrue(done)

    # def test_runtimes(self):
    #     env: OffloadingEnv = gym.make(  # type: ignore
    #             "Offloading-v0",
    #             **{
    #                 'use_raw_state': True,
    #                 'tasks_per_app': 20
    #             },
    #         )
    #     tasks = task_minimum_runtimes()