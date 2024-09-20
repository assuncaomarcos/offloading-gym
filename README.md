# Task Offloading Simulation Environments

## Overview

This project provides a collection of Gymnasium environments designed to simulate various task offloading scenarios.
These environments are useful for training and evaluating reinforcement learning agents in task scheduling, resource
allocation, and load balancing tasks.

## Features

- Gymnasium environments tailored for different offloading scenarios
- Customizable task parameters
- Support for multiple reinforcement learning frameworks

## Installation

To use these environments, you'll need to have Python 3.7+ and some additional packages installed. You can install the
required dependencies via pip:

```bash
pip install -r requirements.txt
```

## Environments

The project includes different environments, each representing a unique task offloading scenario. Below is a brief
overview of the included environments:

### BinaryOffload

A Gymnasium environment for training DRL algorithms to offload tasks from local devices to edge servers. Tasks are
represented as Directed Acyclic Graphs (DAGs) with dynamic workload generation. Based on the research papers:

- Wang, Jin et al. "Dependent task offloading for edge computing based on deep reinforcement learning." IEEE
  Transactions on Computers 71, no. 10 (2021): 2449-2461.
- Wang, Jin et al. "Fast adaptive task offloading in edge computing based on meta reinforcement learning." IEEE
  Transactions on Parallel and Distributed Systems 32, no. 1 (2020): 242-253.
- H. Arabnejad and J. Barbosa. "List Scheduling Algorithm for Heterogeneous Systems by an Optimistic Cost Table." IEEE
  Transactions on Parallel and Distributed Systems, Vol. 25, No. 3, March 2014.

This environment simulates a user device connected to an edge server, allowing task offloading through a network with
distinct upload and download capacities. It implements the scenario used in:

- Wang, Jin et al. "Fast adaptive task offloading in edge computing based on meta reinforcement learning." IEEE
  Transactions on Parallel and Distributed Systems 32, no. 1 (2020): 242-253.

### FogPlacement

An environment that simulates a fog infrastructure comprising mobile devices, edge servers, and cloud servers.
Applications are also structured as DAGs and the workload resembles that of the `BinaryOffload`. The environment
focuses on the placement and execution of tasks onto available resources. Actions involve selecting
which resource (mobile device, edge server, or cloud server) will execute the current task. This is an extended version
of the environment used in:

- Goudarzi, Mohammad et al. "$\mu$-DDRL: A QoS-Aware Distributed Deep Reinforcement Learning Technique for Service
  Offloading in Fog Computing Environments." *IEEE Transactions on Services Computing*, vol. 17, no. 1, pp. 47-59, 2024.

## Usage

To use these environments with `gymnasium`, you need to import Gymnasium and the `offloading.envs` module. You can then
create an instance using the following code snippet:

```python
import gymnasium as gym
import offloading_gym.envs

RNG_SEED = 42

# Create an instance of the environment
env = gym.make("FogPlacement-v0")
obs, info = env.reset(seed=RNG_SEED)

terminated = False
while not terminated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

## Contributing

We welcome contributions to improve and expand the environments. Please fork the repository, create a new branch, and
submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

We would like to thank all contributors and the community for their support and feedback.

