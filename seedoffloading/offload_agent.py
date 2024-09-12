import click
import gymnasium as gym
import offloading_gym as off_gym
from offloading_gym.envs.base import BaseOffEnv
import random
import numpy as np
import tensorflow as tf
import keras
from offloading_gym.config import ModelConfig
from collections import namedtuple
from typing import List
from threading import Thread
from offloading_gym.networks import tpto
from offloading_gym.networks.tpto import Encoder
from pytorch_seed_rl.pytorch_seed_rl.agents.actor import Actor
from pytorch_seed_rl.pytorch_seed_rl.agents.learner import Learner 


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
    # np.random.seed(seed)
    # tf.random.set_seed(seed)


#It creates and initializes a vectorized environment using gymnasium. 
# This environment runs multiple copies of the same environment 
# independently and in parallel by default
def create_vector_env(env_name: str, num_envs: int, seed: int, **kwargs):
    envs = gym.make_vec(env_name, num_envs=num_envs, **kwargs)
    
    envs.reset(seed=seed)
    return envs

def create_experiences(
    env: gym.vector.VectorEnv,
    seed: int,
    trajectories_per_worker: int,
    num_steps: int,
    model: tf.keras.Model
) -> List[Experience]:
    trajectories: List[Experience] = []

    with click.progressbar(
            length=trajectories_per_worker,
            label=f'Creating {trajectories_per_worker * num_steps} experiences'
    ) as bar:
        for traj in bar:
            observations, _ = env.reset(seed=seed + traj)
          
          
            servers_batch = tf.expand_dims(observations["servers"], axis=0)
            tasks_batch = tf.expand_dims(observations["task"], axis=0)

            for step in range(num_steps):
            
                actions, values = model({'input_1': servers_batch, 'input_2': tasks_batch})
                actions = tf.argmax(actions, axis=-1).numpy()
                
            
                observations, rewards, _, _, info = env.step(actions)
               

             
                servers_batch = tf.expand_dims(observations["servers"], axis=0)
                tasks_batch = tf.expand_dims(observations["task"], axis=0)
                
                # Store the experience
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
#@click.option('--env-name', help='Gymnasium environment to use', default='BinaryOffload-v0')
@click.option('--env-name', help='Gymnasium environment to use', default='FogPlacement-v0')
@click.option('--episode-length', help='Number of time steps in an episode', default=EPISODES_PER_TRAJECTORY)
@click.option('--trajectories', help='Number of trajectories per worker and batch', default=TRAJECTORIES_PER_BATCH)
@click.option('--tasks-per-app', help='Number of tasks per mobile application', default=TASKS_PER_APP)

def train(ctx, **options):
    debug = ctx.obj["debug"]
    seed = options['seed']
    set_seed(seed)
    config = ModelConfig()

    if debug:
        options['trajectories'] = 1
        options['episode_length'] = 1
    
    
    # Initialize the environment
    v_env: BaseOffEnv = create_vector_env(
        env_name=options['env_name'],
        num_envs=options['workers'],
        seed=seed,
        tasks_per_app=options['tasks_per_app']
    )
    
    
    # Initialize the Transformer model
    observation_input_1 = tf.keras.Input(shape=(57, 8), dtype=tf.float32, name='input_1')
    observation_input_2 = tf.keras.Input(shape=(20), dtype=tf.float32, name='input_2') 

    encoder = Encoder(config.h, config.d_k, config.d_v, config.d_model, config.d_ff, config.n, config.dropout_rate)
    logits, value = encoder([observation_input_1, observation_input_2], training=True, mask=None)
    model = tf.keras.Model(inputs=[observation_input_1, observation_input_2], outputs=[logits, value])

    #initialize the Learner
    #rank for learner is 0, for actor can be greater than 0
    
    num_env = options['workers']
    learner = Learner(rank = 0, num_actors= 2, env = v_env, num_env = num_env ,model= model, optimizer= config.policy_optimizer)
   
    actor = Actor(
        rank = 1,
        learner = learner,
        env=v_env,
        env_name=options['env_name'],
        num_envs=options['workers'],
        seed=seed,
        tasks_per_app=options['tasks_per_app']
    )
     
    # Training loop
    for epoch in range(1, options['epochs'] + 1):
        seed_for_epoch = seed + epoch
        print(f'Current epoch: {epoch}')

        # Actor loop for generating experince/model inference 
        actor._loop()

        # Generate experiences using the Actor
        # experiences: List[Experience] = create_experiences(
        #     env=v_env,
        #     seed=seed_for_epoch,
        #     trajectories_per_worker=options['trajectories'],
        #     num_steps=options['episode_length'],
        #     model=model
        #)

if __name__ == '__main__':
    main(obj={})









