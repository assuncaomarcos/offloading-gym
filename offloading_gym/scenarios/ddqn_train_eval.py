#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DDQN using actor/learner in the offloading environment.

To run this training:

```bash
tensorboard --logdir $HOME/tmp/ddqn_offloading --port 2223 &
python -m offloading_gym.scenarios.ddqn_train_eval.py --root_dir=$HOME/tmp/ddqn_offloading
```

"""

import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin
import reverb
import gymnasium as gym
from offloading_gym.envs.wrappers import MultiBinaryToDiscreteWrapper, GymnasiumWrapper
from tensorflow import keras
from tf_agents.agents.dqn import dqn_agent
from tf_agents.metrics import py_metrics
from tf_agents.networks import sequential, q_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import train_utils, spec_utils
from tf_agents.utils import common

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'root_dir',
    os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
    'Root directory for writing logs/summaries/checkpoints.',
)
flags.DEFINE_integer(
    'reverb_port',
    None,
    'Port for reverb server, if None, use a randomly chosen unused port.',
)
flags.DEFINE_integer(
    'num_epochs', 100, 'Total number train/eval iterations to perform.'
)
flags.DEFINE_integer(
    'eval_interval',
    300,
    'Number of train steps between evaluations. Set to 0 to skip.',
)
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')


@gin.configurable
def train_eval(
        root_dir,
        # Environment parameters
        tasks_per_app=20,
        env_name="BinaryOffload-v0",
        max_episode_steps=1,
        weight_latency=1.0,
        weight_energy=0.0,
        normalize_task_ids=True,
        # Training params
        num_epochs=100,
        episodes_per_epoch=128,
        training_iterations=100,
        fc_layer_params=(128, 64),
        # Agent params
        epsilon_greedy=0.02,
        batch_size=100,
        dropout_params=(0.1, 0.1),
        learning_rate=5e-4,
        n_step_update=1,
        gamma=0.99,
        target_update_tau=0.05,
        target_update_period=5,
        reward_scale_factor=1.0,
        # Replay params
        reverb_port=None,
        replay_capacity=10000,
        # Others
        policy_save_interval=50,
        eval_interval=300,
        eval_episodes=128,
):
    """Trains and evaluates DQN."""

    def create_environment():
        return GymnasiumWrapper(
            MultiBinaryToDiscreteWrapper(
                gym.make(
                    id=env_name,
                    **{
                        "tasks_per_app": tasks_per_app,
                        "max_episode_steps": max_episode_steps,
                        "weight_latency" : weight_latency,
                        "weight_energy" : weight_energy,
                        "normalize_task_ids": normalize_task_ids
                    },
                )
            )
        )

    collect_env = create_environment()
    eval_env = create_environment()

    observation_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
        spec_utils.get_tensor_specs(collect_env)
    )

    train_step = train_utils.create_train_step()
    q_net = q_network.QNetwork(
        observation_tensor_spec,
        action_tensor_spec,
        fc_layer_params=fc_layer_params,
        kernel_initializer=keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03
        ),
        dropout_layer_params=dropout_params
    )

    agent = dqn_agent.DqnAgent(
        time_step_tensor_spec,
        action_tensor_spec,
        q_network=q_net,
        epsilon_greedy=epsilon_greedy,
        n_step_update=n_step_update,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        td_errors_loss_fn=common.element_wise_huber_loss,
        # td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step,
    )

    agent.initialize()

    table_name = 'uniform_table'
    sequence_length = n_step_update + 1
    table = reverb.Table(
        table_name,
        max_size=replay_capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )
    reverb_server = reverb.Server([table], port=reverb_port)
    reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        sequence_length=sequence_length,
        table_name=table_name,
        local_server=reverb_server,
    )
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        reverb_replay.py_client,
        table_name,
        sequence_length=sequence_length,
        stride_length=1,
    )

    dataset = reverb_replay.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, # num_steps=2
    ).prefetch(3)
    experience_dataset_fn = lambda: dataset

    saved_model_dir = os.path.join(root_dir, learner.POLICY_SAVED_MODEL_DIR)
    env_step_metric = py_metrics.EnvironmentSteps()

    learning_triggers = [
        triggers.PolicySavedModelTrigger(
            saved_model_dir,
            agent,
            train_step,
            interval=policy_save_interval,
            metadata_metrics={triggers.ENV_STEP_METADATA_KEY: env_step_metric},
        ),
        triggers.StepPerSecondLogTrigger(train_step, interval=training_iterations),
    ]

    dqn_learner = learner.Learner(
        root_dir,
        train_step,
        agent,
        experience_dataset_fn,
        triggers=learning_triggers,
    )

    # If we haven't trained yet make sure we collect some random samples first to
    # fill up the Replay Buffer with some experience.
    random_policy = random_py_policy.RandomPyPolicy(
        collect_env.time_step_spec(), collect_env.action_spec()
    )
    initial_collect_actor = actor.Actor(
        collect_env,
        random_policy,
        train_step,
        steps_per_run=1,
        episodes_per_run=episodes_per_epoch,
        observers=[rb_observer],
    )
    logging.info('Doing initial collect.')
    initial_collect_actor.run()

    tf_collect_policy = agent.collect_policy
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_collect_policy, use_tf_function=True
    )

    collect_actor = actor.Actor(
        collect_env,
        collect_policy,
        train_step,
        steps_per_run=1,
        episodes_per_run=episodes_per_epoch,
        observers=[rb_observer, env_step_metric],
        metrics=actor.collect_metrics(10),
        summary_dir=os.path.join(root_dir, learner.TRAIN_DIR),
    )

    tf_greedy_policy = agent.policy
    greedy_policy = py_tf_eager_policy.PyTFEagerPolicy(
        tf_greedy_policy, use_tf_function=True
    )

    eval_actor = actor.Actor(
        eval_env,
        greedy_policy,
        train_step,
        steps_per_run=1,
        episodes_per_run=eval_episodes,
        metrics=actor.eval_metrics(eval_episodes),
        summary_dir=os.path.join(root_dir, 'eval'),
    )

    if eval_interval:
        logging.info('Evaluating.')
        eval_actor.run_and_log()

    logging.info('Training.')
    for _ in range(num_epochs):
        collect_actor.run()
        dqn_learner.run(iterations=training_iterations)
        if eval_interval and dqn_learner.train_step_numpy % eval_interval == 0:
            logging.info('Evaluating.')
            eval_actor.run_and_log()

    rb_observer.close()
    reverb_server.stop()


def main(_):
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings)

    train_eval(
        FLAGS.root_dir,
        num_epochs=FLAGS.num_epochs,
        reverb_port=FLAGS.reverb_port,
        eval_interval=FLAGS.eval_interval,
    )


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    multiprocessing.handle_main(functools.partial(app.run, main))
