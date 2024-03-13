import unittest
import gymnasium as gym
from ..envs.offloading import OffloadingEnv


class TestOffloadingEnv(unittest.TestCase):
    def test_instantiation_with_gym(self):
        gym.make('Offloading-v0')

    def test_environment_with_trivial_agent(self):
        env: OffloadingEnv = gym.make(  # type: ignore
            "Offloading-v0",
            **{
                'use_raw_state': True,
                'tasks_per_app': 20
            },
        )
        env.seed(5)
        _ = env.reset()
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