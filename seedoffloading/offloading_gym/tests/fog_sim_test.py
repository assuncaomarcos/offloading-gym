import unittest
import networkx as nx
from ..simulation.fog import ComputeResource, ComputingEnvironment, typing
from ..task_graph import TaskAttr, EdgeAttr, TaskGraph
import simpy

CPU_CORE_FREQUENCY = 1.0  # 1 GHz
CPU_CORES = 8
MEMORY = 1.0  # 1GB


class TestSimulation(unittest.TestCase):

    def setUp(self) -> None:
        self.env = simpy.Environment()
        self.resource = ComputeResource(
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
    def execute_task(resource: ComputeResource, n_cores: int = 1, memory: float = 0.1):
        """Simulates task execution using simpy"""
        with resource.request(cpu_cores=n_cores, memory=memory) as req:
            yield req
            # start_time = resource.env.now
            yield resource.env.timeout(5.0)
            # finish_time = resource.env.now

    def test_create_infra(self):
        fog_env = ComputingEnvironment.build(
            simpy_env=self.env, seed=42, config=typing.DEFAULT_COMP_CONFIG
        )
        self.assertEqual(len(fog_env.comp_resources), 57)
        iot_device = fog_env.comp_resources[0]
        self.assertEqual(iot_device.resource_id, 0)
        self.assertEqual(iot_device.resource_type, typing.ResourceType.IOT)
        print(fog_env.net_resources[0][3])
