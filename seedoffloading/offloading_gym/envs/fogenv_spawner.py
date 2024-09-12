from typing import List
import gym
from offloading_gym.envs.fog.env import FogPlacementEnv


class EnvSpawner():
 

    def __init__(self, env_id: str, num_envs: int = 1):
        # ATTRIBUTES
        self.env_id = env_id
        self.num_envs = num_envs
        self._generate_env_info()

    def spawn(self) -> List[gym.Env]:
        """Returns a list of offloading environments."""
        return [FogPlacementEnv(env_id=self.env_id) for _ in range(self.num_envs)]

    def _generate_env_info(self):
        """Spawns environment once to save properties for later reference by learner and model."""
        placeholder_env = self.spawn()[0]

        self.env_info = {
            "env_id": self.env_id,
            "num_envs": self.num_envs,
            "action_space": placeholder_env.action_space,
            "observation_space": placeholder_env.observation_space,
            "reward_range": placeholder_env.reward_range,
           # "max_episode_steps": placeholder_env.spec.max_episode_steps
        }

        self.placeholder_obs = placeholder_env.reset()

        placeholder_env.close()
        del placeholder_env
