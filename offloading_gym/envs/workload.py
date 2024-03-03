#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..workload import daggen, distribution as dt
from typing import Optional, List
from networkx import DiGraph


class DaggenWorkload(dt.DistributionWorkload):

    def __init__(self, length=0):
        super().__init__(length)

    def step(self, offset=1) -> List[Optional[DiGraph]]:
        pass

    def __len__(self):
        return self.length

    def peek(self):
        return self.step(0)
