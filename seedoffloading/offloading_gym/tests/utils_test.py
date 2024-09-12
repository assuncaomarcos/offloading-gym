import unittest
import numpy as np
import offloading_gym.utils.arrays as arrays


class TestUtils(unittest.TestCase):

    def test_pad_list(self):
        lst = [1, 2, 3, 4, 5]
        padded = arrays.pad_list(lst=lst, target_length=10)
        self.assertEqual(len(padded), 10)
        self.assertListEqual(padded[5:], [-1.0, -1.0, -1.0, -1.0, -1.0])
        trimmed = arrays.pad_list(lst=padded, target_length=3)
        self.assertListEqual(trimmed, [1, 2, 3])

    def test_pad_array(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        padded = arrays.pad_array(arr=arr, target_length=10, pad_value=-2.0)
        self.assertEqual(len(padded), 10)
        self.assertTrue(np.all(padded[5:] == -2.0))
        trimmed = arrays.pad_list(lst=padded, target_length=3)
        self.assertTrue(np.array_equal(trimmed, np.array([1.0, 2.0, 3.0])))

    def test_binary_sequences(self):
        length = 4
        seq = arrays.binary_sequences(length)
        self.assertEqual(len(seq), length * length)
        self.assertEqual(seq[0], (0, 0, 0, 0))
        self.assertEqual(seq[length * length - 1], (1, 1, 1, 1))
