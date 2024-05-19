#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import gymnasium as gym
import offloading_gym as off_gym
import random
import numpy as np

from collections import namedtuple
from typing import List
from offloading_gym.networks import tpto

NUMBER_EPOCHS = 10
PARALLEL_WORKERS = 5
TASKS_PER_APP = 20
TRAJECTORIES_PER_BATCH = 100
EPISODES_PER_TRAJECTORY = 5
LEARNING_RATE = 1e-2

Experience = namedtuple(
    'Experience',
    field_names='embeddings actions rewards task_rewards task_energy'.split()
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


def create_experiences(
        env: gym.vector.VectorEnv,
        seed: int,
        trajectories_per_worker: int,
        num_steps: int,
        model: object
) -> List[Experience]:
    trajectories: List[Experience] = []

    with click.progressbar(
            length=trajectories_per_worker,
            label=f'Creating {trajectories_per_worker * num_steps} experiences'
    ) as bar:
        for traj in bar:
            observations, _ = env.reset(seed=seed + traj)

            for step in range(num_steps):
                actions = env.action_space.sample()
                observations, rewards, _, _, info = env.step(actions=actions)
                experience = Experience(
                    embeddings=observations,
                    actions=actions,
                    rewards=rewards,
                    task_rewards=info["task_rewards"],
                    task_energy=info["task_energy"]
                )
                trajectories.append(experience)

    # Shuffle the experience list to avoid bias
    return random.shuffle(trajectories)


@main.command()
@click.pass_context
@click.option('--epochs', help='Number of epochs to train', default=NUMBER_EPOCHS)
@click.option('--workers', help='Number of workers to use', default=PARALLEL_WORKERS)
@click.option('--seed', help='Random seed to use', default=42)
@click.option('--lr', help='Learning rate', default=LEARNING_RATE)
@click.option('--env-name', help='Gymnasium environment to use', default='BinaryOffload-v0')
@click.option('--episode-length', help='Number of time steps in an episode', default=EPISODES_PER_TRAJECTORY)
@click.option('--trajectories', help='Number of trajectories per worker and batch', default=TRAJECTORIES_PER_BATCH)
@click.option('--tasks-per-app', help='Number of tasks per mobile application', default=TASKS_PER_APP)
def train(ctx, **options):
    debug = ctx.obj["debug"]
    seed = options['seed']
    set_seed(seed)

    if debug:
        options['trajectories'] = 1
        options['episode_length'] = 1

    v_env: off_gym.BaseOffEnv = create_vector_env(  # type: ignore
        env_name=options['env_name'],
        num_envs=options['workers'],
        seed=seed,
        tasks_per_app=options['tasks_per_app']
    )

    model = tpto.TransformerPPO(env=v_env)

    for epoch in range(1, options['epochs'] + 1):
        seed_for_epoch = seed + epoch
        print(f'Current epoch: {epoch}')

        experiences: List[Experience] = create_experiences(
            env=v_env,
            seed=seed_for_epoch,
            trajectories_per_worker=options['trajectories'],
            num_steps=options['episode_length'],
            model=model
        )


if __name__ == '__main__':
    main(obj={})
