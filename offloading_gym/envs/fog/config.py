#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Default environment configuration"""

from offloading_gym.simulation.fog.typing import (
    ComputingConfig,
    WorkloadConfig,
    Coordinate,
    RectGeographicalArea,
    ResourceConfig,
    ResourceGroupConfig,
    NetworkConfig,
    Interval,
)

import math

from .cloud import cloud_sites
from ...simulation.fog.energy import JingEdgeEnergyModel, JingIoTEnergyModel

__all__ = ["DEFAULT_COMP_CONFIG", "DEFAULT_WORKLOAD_CONFIG"]

BYTES_PER_MEGABYTE = 2 ** 20

# The following GPS coordinates roughly encompass the entire
# city of Montreal, forming a rough rectangular boundary around it.
MONTREAL_AREA = RectGeographicalArea(
    northeast=Coordinate(lat=45.7057, long=-73.4746),
    northwest=Coordinate(lat=45.7057, long=-73.9434),
    southeast=Coordinate(lat=45.3831, long=-73.4746),
    southwest=Coordinate(lat=45.3831, long=-73.9434),
)

DEFAULT_COMP_CONFIG = ComputingConfig(
    iot=ResourceGroupConfig(
        num_resources=1,
        resource_config=ResourceConfig(
            cpu_cores=[1],
            cpu_core_speed=[1.0],
            memory=[1.0],
            energy_model=(
                JingIoTEnergyModel,
                {"energy_coefficient": math.pow(10, -27)},
            ),
        ),
        network_config=None,
        deployment_area=MONTREAL_AREA,
    ),
    edge=ResourceGroupConfig(
        num_resources=36,
        resource_config=ResourceConfig(
            cpu_cores=[4],
            cpu_core_speed=[1.5, 1.8, 2.0],
            memory=[1.0, 2.0, 4.0],
            energy_model=(
                JingEdgeEnergyModel,
                {"energy_per_unit": 9 * math.pow(10, -5)},
            ),
        ),
        network_config=NetworkConfig(
            bandwidth=Interval(min=10, max=12), propagation_speed=3 * 10 ** 8
        ),
        deployment_area=MONTREAL_AREA,
    ),
    cloud=ResourceGroupConfig(
        num_resources=20,
        resource_config=ResourceConfig(
            cpu_cores=[8],
            cpu_core_speed=[2.0, 2.6, 3.0],
            memory=[16.0, 24.0, 32.0],
            energy_model=(
                JingEdgeEnergyModel,
                {"energy_per_unit": 9 * math.pow(10, -5)},
            ),
        ),
        network_config=NetworkConfig(
            bandwidth=Interval(min=4, max=8), propagation_speed=2.07 * 10 ** 8
        ),
        deployment_area=cloud_sites(),
    ),
)

DEFAULT_WORKLOAD_CONFIG = WorkloadConfig(
    num_tasks=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    min_computing=(10 ** 7),
    max_computing=(3 * 10 ** 8),
    min_memory=25 * BYTES_PER_MEGABYTE,
    max_memory=100 * BYTES_PER_MEGABYTE,
    # Each task produces between 50KB and 200KB of data
    min_datasize=51200,
    max_datasize=204800,
    density_values=[0.4, 0.5, 0.6, 0.7, 0.8],
    regularity_values=[0.2, 0.5, 0.8],
    fat_values=[0.4, 0.5, 0.6, 0.7, 0.8],
    ccr_values=[0.3, 0.4, 0.5],
    jump_values=[1, 2],
)
