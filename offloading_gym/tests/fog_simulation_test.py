import unittest
import networkx as nx
from ..simulation.fog import ComputeResource
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
            memory_capacity=MEMORY
        )

    def test_ok_requests(self):
        for _ in range(CPU_CORES * 2):
            self.env.process(self.execute_task(self.resource))
        self.env.run()

    def test_not_ok_cpus(self):
        self.env.process(self.execute_task(self.resource, n_cores=CPU_CORES + 1))
        self.assertRaises(simpy.exceptions.SimPyException, self.env.run)

    def test_not_ok_memory(self):
        self.env.process(self.execute_task(self.resource, n_cores=CPU_CORES, memory=MEMORY + 0.1))
        self.assertRaises(simpy.exceptions.SimPyException, self.env.run)

    def execute_task(self, resource: ComputeResource, n_cores: int = 1, memory: float = 0.1):
        with resource.request(cpu_cores=n_cores, memory=memory) as req:
            yield req
            start_time = resource.env.now
            yield resource.env.timeout(5.0)
            finish_time = resource.env.now
            print(req, start_time, finish_time)