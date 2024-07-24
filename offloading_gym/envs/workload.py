#!/usr/bin/env python
# -*- coding: utf-8 -*-

from offloading_gym.workload import RandomDAGGenerator

__all__ = ["RandomGraphWorkload"]


class RandomGraphWorkload(RandomDAGGenerator):

    def __init__(self, length: int = 0, **kwargs):
        super().__init__(length, **kwargs)

    @classmethod
    def build(cls, args: dict):
        kwargs = {k: v for k, v in args.items()}
        return RandomGraphWorkload(**kwargs)

