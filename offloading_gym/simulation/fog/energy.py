#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Models for computing the energy consumed by resources."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from offloading_gym.envs.fog.workload import FogTaskAttr
    from offloading_gym.simulation.fog.resources import GeolocatedResource

import math

__all__ = [
    "EnergyModel",
    "LinearEnergyModel",
    "JingIoTEnergyModel",
    "JingEdgeEnergyModel",
]


class EnergyModel(ABC):

    @abstractmethod
    def energy_use(
            self, resource: "GeolocatedResource", task: "FogTaskAttr"
    ) -> float: ...

    """ Computes the energy consumed by the resource to execute the task. """


@dataclass
class LinearEnergyModel(EnergyModel):
    """
    Linear energy model for computing resources.

    Attributes:
        idle_power (float): The idle power consumption.
        max_power (float): The maximum power consumption.

    """

    idle_power: float
    max_power: float

    def energy_use(self, resource: "GeolocatedResource", task: "FogTaskAttr") -> float:
        return (self.idle_power * task.makespan) + (self.max_power * task.runtime)


class JingIoTEnergyModel(EnergyModel):
    """Energy model for IoT devices according to the paper by Jing et al.:

    Jiang, Hongbo, Xingxia Dai, Zhu Xiao, and Arun Iyengar. Joint task offloading
    and resource allocation for energy-constrained mobile edge computing.
    IEEE Transactions on Mobile Computing 22, no. 7 (2022): 4000-4015.
    """

    energy_coefficient: float

    def __init__(
            self,
            energy_coefficient: float = math.pow(10, -27),
    ):
        self.energy_coefficient = energy_coefficient

    def energy_use(self, resource: "GeolocatedResource", task: "FogTaskAttr") -> float:
        return (
                self.energy_coefficient
                * task.processing_demand
                * math.pow(resource.cpu_core_speed * resource.number_of_cores, 2)
        )


class JingEdgeEnergyModel(EnergyModel):
    """Energy model of edge servers according to the paper by Jing et al.:

    Jiang, Hongbo, Xingxia Dai, Zhu Xiao, and Arun Iyengar. Joint task offloading
    and resource allocation for energy-constrained mobile edge computing.
    IEEE Transactions on Mobile Computing 22, no. 7 (2022): 4000-4015.
    """

    energy_per_unit: float

    def __init__(self, energy_per_unit: float = 9 * math.pow(10, -5)):
        self.energy_per_unit = energy_per_unit

    def energy_use(self, resource: "GeolocatedResource", task: "FogTaskAttr") -> float:
        return self.energy_per_unit * task.runtime
