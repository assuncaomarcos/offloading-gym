#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Any


def pad_list(lst: List[Any], target_length: int, pad_value: float = -1.0):
    padding = target_length - len(lst)
    if padding > 0:
        lst.extend([pad_value] * padding)
    return lst


def pad_array(arr: np.ndarray, target_length: int, pad_value: float = -1.0):
    padding = target_length - len(arr)
    if padding > 0:
        arr = np.pad(arr, (0, padding), 'constant', constant_values=(pad_value,))
    return arr


def merge_arrays(arrays, to_numpy_array=True):
    merged = []
    for arr in arrays:
        merged += arr.tolist() if hasattr(arr, 'tolist') else arr
    return np.array([merged]) if to_numpy_array else [merged]


def create_binary_sequences(length):
    """
    Generates binary sequences of a given length.

    Args:
        length (int): The length of the binary sequences to generate.

    Returns:
        list: A list of binary sequences, where each sequence is represented as a list of integers.

    Example:
        >>> create_binary_sequences(3)
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    """
    sequences = []
    for i in range(2 ** length):
        binary_str = bin(i)[2:]
        binary_sequence = [int(x) for x in binary_str]
        while len(binary_sequence) < length:
            binary_sequence.insert(0, 0)
        sequences.append(binary_sequence)
    return sequences


