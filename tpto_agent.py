#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import gymnasium as gym
import offloading_gym as off_gym

NUMBER_ITERATIONS = 10
PARALLEL_WORKERS = 5
TASKS_PER_APP = 20
TRAJECTORIES_PER_BATCH = 100


@click.group()
@click.pass_context
@click.option('--debug/--no-debug', default=False)
def main(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


def create_vector_env(env_name: str, num_envs: int, seed: int, **kwargs):
    envs = gym.make_vec(env_name, num_envs=num_envs, **kwargs)
    envs.reset(seed=seed)
    return envs


@main.command()
@click.pass_context
@click.option('--epochs', help='Number of epochs to train', default=NUMBER_ITERATIONS)
@click.option('--workers', help='Number of workers to train', default=PARALLEL_WORKERS)
@click.option('--seed', help='Random seed to use', default=42)
@click.option('--lr', help='Learning rate', default=1e-2)
@click.option('--env-name', help='Gymnasium environment to use', default='BinaryOffload-v0')
@click.option('--episode-length', help='Number of time steps in an episode', default=1)
@click.option('--trajectories', help='Number of trajectories per batch', default=TRAJECTORIES_PER_BATCH)
@click.option('--tasks-per-app', help='Number of tasks per mobile application', default=TASKS_PER_APP)
def train(ctx, epochs, workers, seed, lr, env_name, episode_length, trajectories, tasks_per_app):

    envs = create_vector_env(
        env_name=env_name,
        num_envs=workers,
        seed=seed,
        tasks_per_app=tasks_per_app
    )


if __name__ == '__main__':
    main(obj={})
