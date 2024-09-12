import unittest


class TestSimulation(unittest.TestCase):

    def test_module_import(self):
        """Tests importing the backbone module."""
        from offloading_gym.simulation.fog import backbone
        self.assertEqual(backbone.server_info()[0].title, 'Toronto')
