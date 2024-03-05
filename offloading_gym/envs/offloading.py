#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import BaseOffEnv


class OffloadingEnv(BaseOffEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def state(self):
        pass
