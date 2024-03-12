#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..scheduler import Scheduler

DEFAULT_SCHEDULER_CONFIG = {
    "type": "fifo_scheduler",
    "num_edge_resources": 10,
    "edge_resource_fps": 1024 * 1024,
    "num_user_devices": 1,
    "user_device_fps": 1024 * 1024,
    "comm_link_mbps": 10
}


class DefaultScheduler(Scheduler):

    def schedule_tasks(self, tasks):
        pass

    @staticmethod
    def build(**kwargs):
        return DefaultScheduler(**kwargs)


def build_scheduler(scheduler_config: dict):
    scheduler_type = scheduler_config['type']
    kwargs = {k: v for k, v in scheduler_config.items() if k != 'type'}
    if scheduler_type == 'fifo_scheduler':
        return DefaultScheduler.build(**kwargs)
    else:
        raise RuntimeError(f'Unsupported scheduler type {scheduler_type}')
