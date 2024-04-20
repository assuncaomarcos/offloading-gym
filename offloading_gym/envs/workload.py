#!/usr/bin/env python
# -*- coding: utf-8 -*-

from offloading_gym.workload import RandomDAGGenerator

__all__ = ["build_workload"]


class RandomGraphWorkload(RandomDAGGenerator):

    def __init__(self, length: int = 0, **kwargs):
        super().__init__(length, **kwargs)

    @staticmethod
    def build(**kwargs):
        return RandomGraphWorkload(**kwargs)


def build_workload(workload_config: dict):
    wkl_type = workload_config["type"]
    kwargs = {k: v for k, v in workload_config.items() if k != "type"}
    if wkl_type == "random_dag":
        return RandomGraphWorkload.build(**kwargs)
    else:
        raise RuntimeError(f"Unsupported workload type {wkl_type}")
