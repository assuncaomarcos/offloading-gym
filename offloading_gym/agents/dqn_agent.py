import gin
import tensorflow as tf
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent


@gin.configurable
class D3qnAgent(dqn_agent.DqnAgent):
    """A Dueling DQN Agent.

    Implements the Double Dueling DQN algorithm from

    "Dueling Network Architectures for Deep Reinforcement Learning"
     Wang et al., 2016
     https://arxiv.org/abs/1511.06581
    """

    def _compute_next_q_values(self, next_time_steps, info):
        """Compute the q value of the next state for TD error computation.

        Args:
          next_time_steps: A batch of next timesteps
          info: PolicyStep.info that may be used by other agents inherited from
            dqn_agent.

        Returns:
          A tensor of Q values for the given next state.
        """
        del info
        # TODO(b/117175589): Add binary tests for DDQN.
        network_observation = next_time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation
            )

        q_next_target, _ = self._target_q_network(
            network_observation, step_type=next_time_steps.step_type
        )
        next_target_q_values = (
            q_next_target[0] if isinstance(q_next_target, tuple) else q_next_target
        )
        q_next, _ = self._q_network(
            network_observation, step_type=next_time_steps.step_type
        )
        next_q_values = q_next[1] if isinstance(q_next, tuple) else q_next
        best_next_actions = tf.math.argmax(next_q_values, axis=1)

        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        multi_dim_actions = tf.nest.flatten(self._action_spec)[0].shape.rank > 0
        return common.index_with_actions(
            next_target_q_values,
            best_next_actions,
            multi_dim_actions=multi_dim_actions,
        )
