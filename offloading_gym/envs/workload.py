#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..workload import RandomDAGGenerator

__all__ = [
    "build_workload",
    "RANDOM_WORKLOAD_CONFIG"
]

RANDOM_WORKLOAD_CONFIG = {
    "type": "random_dag",
    "num_tasks": 20,
    "min_computing": 10 ** 7,           # Each task requires between 10^7 and 10^8 cycles
    "max_computing": 10 ** 8,
    "min_datasize": 5120,               # Each task produces between 5KB and 50KB of data
    "max_datasize": 51200,
    "density_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "regularity_values": [0.2, 0.5, 0.8],
    "fat_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "ccr_values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "jump_values": [1, 2, 4],
}


class RandomGraphWorkload(RandomDAGGenerator):

    def __init__(self, length: int = 0, **kwargs):
        super().__init__(length, **kwargs)

    @staticmethod
    def build(**kwargs):
        return RandomGraphWorkload(**kwargs)


def build_workload(workload_config: dict):
    wkl_type = workload_config['type']
    kwargs = {k: v for k, v in workload_config.items() if k != 'type'}
    if wkl_type == 'random_dag':
        return RandomGraphWorkload.build(**kwargs)
    else:
        raise RuntimeError(f'Unsupported workload type {wkl_type}')
