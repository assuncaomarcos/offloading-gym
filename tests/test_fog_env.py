import unittest
from offloading_gym.simulation.fog import resources, typing, ComputingEnvironment
from offloading_gym.envs.fog import config
import simpy
import gymnasium as gym
import numpy as np

CPU_CORE_FREQUENCY = 1.0  # 1 GHz
CPU_CORES = 8
MEMORY = 1.0  # 1GB
RNG_SEED = 42


class TestFogSimulation(unittest.TestCase):

    def setUp(self) -> None:
        self.env = simpy.Environment()
        self.resource = resources.ComputeResource(
            env=self.env,
            n_cpu_cores=CPU_CORES,
            cpu_core_speed=CPU_CORE_FREQUENCY,
            memory_capacity=MEMORY,
        )

    def test_ok_requests(self):
        """Tests a few requests that respect the maximum resource capacity."""
        for _ in range(CPU_CORES * 2):
            self.env.process(self.execute_task(self.resource))
        self.env.run()

    def test_not_ok_cpus(self):
        """Tests a few requests that exceed the maximum number of CPU cores."""
        self.env.process(self.execute_task(self.resource, n_cores=CPU_CORES + 1))
        self.assertRaises(simpy.exceptions.SimPyException, self.env.run)

    def test_not_ok_memory(self):
        """Tests a few requests that exceed the maximum amount of memory."""
        self.env.process(
            self.execute_task(self.resource, n_cores=CPU_CORES, memory=MEMORY + 0.1)
        )
        self.assertRaises(simpy.exceptions.SimPyException, self.env.run)

    @staticmethod
    def execute_task(
            resource: resources.ComputeResource, n_cores: int = 1, memory: float = 0.1
    ):
        """Simulates task execution using simpy"""
        with resource.request(cpu_cores=n_cores, memory=memory) as req:
            yield req
            yield resource.env.timeout(5.0)

    def test_create_infra(self):
        fog_env = ComputingEnvironment.build(
            simpy_env=self.env, seed=42, config=config.DEFAULT_COMP_CONFIG
        )
        self.assertEqual(len(fog_env.compute_resources), 57)
        iot_device = fog_env.compute_resources[0]
        self.assertEqual(iot_device.resource_id, 0)
        self.assertEqual(iot_device.resource_type, typing.ResourceType.IOT)


class TestFogEnv(unittest.TestCase):

    def setUp(self) -> None:
        self.rng = np.random.default_rng(RNG_SEED)
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

        try:
            for _ in range(50):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    _, _ = env.reset()

        except Exception as error:
            self.fail(f"Unexpected error: {error}")
