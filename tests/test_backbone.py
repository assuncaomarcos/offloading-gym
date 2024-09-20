import unittest

"""Tests the module that loads the latency information obtained from Wonderproxy"""

NUM_CLOUD_SITES = 76  # as of Sep 17, 2024


class TestSimulation(unittest.TestCase):

    def setUp(self):
        from offloading_gym.envs.fog import cloud
        self.cloud_sites = cloud.cloud_sites()

    def test_num_sites(self):
        """Tests length of sites list"""
        self.assertEqual(len(self.cloud_sites), NUM_CLOUD_SITES, "Incorrect number of cloud sites")

    def test_elements(self):
        """Tests a few elements"""
        self.assertEqual(self.cloud_sites[0].title, 'Toronto', "Incorrect site title")
