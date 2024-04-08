#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import gymnasium as gym
import offloading_gym as off_gym
import random
import numpy as np

from collections import namedtuple
from typing import List

NUMBER_EPOCHS = 10
PARALLEL_WORKERS = 5
TASKS_PER_APP = 20
TRAJECTORIES_PER_BATCH = 200
EPISODES_PER_TRAJECTORY = 50
LEARNING_RATE = 1e-2

Experience = namedtuple(
    'Experience',
    field_names='embedding action rewards'.split()
)


@click.group()
@click.pass_context
@click.option('--debug/--no-debug', default=False)
def main(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


def set_seed(seed: int):
    """ Sets a random seed for reproducibility """
    random.seed(seed)
    # tf.random.set_seed(seed)


def create_vector_env(env_name: str, num_envs: int, seed: int, **kwargs):
    envs = gym.make_vec(env_name, num_envs=num_envs, **kwargs)
    envs.reset(seed=seed)
    return envs


def generate_experiences(
        env: gym.vector.VectorEnv,
        seed: int,
        trajectories_per_worker: int,
        num_steps: int,
        model: object
) -> List[Experience]:
    trajectories: List[Experience] = []

    for traj in range(trajectories_per_worker):
        observations, _ = env.reset(seed=seed + traj)

        for step in range(num_steps):
            observations, reward, _, truncate, info = env.step(env.action_space.sample())

    return trajectories


@main.command()
@click.pass_context
@click.option('--epochs', help='Number of epochs to train', default=NUMBER_EPOCHS)
@click.option('--workers', help='Number of workers to train', default=PARALLEL_WORKERS)
@click.option('--seed', help='Random seed to use', default=42)
@click.option('--lr', help='Learning rate', default=LEARNING_RATE)
@click.option('--env-name', help='Gymnasium environment to use', default='BinaryOffload-v0')
@click.option('--episode-length', help='Number of time steps in an episode', default=1)
@click.option('--trajectories', help='Number of trajectories per batch', default=TRAJECTORIES_PER_BATCH)
@click.option('--tasks-per-app', help='Number of tasks per mobile application', default=TASKS_PER_APP)
def train(ctx, epochs, workers, seed, lr, env_name, episode_length, trajectories, tasks_per_app):
    debug = ctx.obj["debug"]
    set_seed(seed)

    v_env: off_gym.BaseOffEnv = create_vector_env(  # type: ignore
        env_name=env_name,
        num_envs=workers,
        seed=seed,
        tasks_per_app=tasks_per_app
    )

    model = None

    for epoch in range(1, epochs + 1):
        seed_for_epoch = seed + epoch
        print(f'Current epoch: {epoch}')
        experiences: List[Experience] = generate_experiences(
            env=v_env,
            seed=seed_for_epoch,
            trajectories_per_worker=trajectories,
            num_steps=episode_length,
            model=model
        )


if __name__ == '__main__':
    main(obj={})
