import collections
import statistics
from typing import Any, List, Tuple
import time

import gym
import numpy as np
import tensorflow as tf
import tqdm

from a2c.model import Model


class Algorithm:
    RENDER_SLEEP_TIME = 0.01
    OTHER_STATE = "otherObs"

    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.99

    REWARD_THRESHOLD = 0.4

    def __init__(self, env: gym.Env, other_model: Any, max_episodes: int, max_steps: int) -> None:
        self.env = env
        self.model = Model(2 ** env.action_space.n if type(env.action_space)
                           is gym.spaces.MultiBinary else env.action_space.n)
        self.other_model = other_model
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.compat.v2.losses.Reduction.SUM)
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.eps = np.finfo(np.float32).eps.item()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=Algorithm.LEARNING_RATE)

    def load_model(self, file_path: str) -> None:
        self.model = tf.keras.models.load_model(file_path)

    def save_model(self, file_path: str) -> None:
        self.model.save(file_path)

    def train(self) -> None:
        min_episodes_criterion = 100

        running_reward = 0

        # Keep last episodes reward
        episodes_reward = collections.deque(maxlen=min_episodes_criterion)

        with tqdm.trange(self.max_episodes) as t:
            for i in t:
                initial_state = tf.constant(self.env.reset(), dtype=tf.float32)
                episode_reward = int(self._train_step(initial_state, Algorithm.DISCOUNT_FACTOR))

                episodes_reward.append(episode_reward)
                running_reward = statistics.mean(episodes_reward)

                t.set_description(f"Episode {i}")
                t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

                # Show average episode reward every 10 episodes
                if i % 10 == 0:
                    pass  # print(f'Episode {i}: average reward: {avg_reward}')

                if running_reward > Algorithm.REWARD_THRESHOLD and i >= min_episodes_criterion:
                    break

        print(f"\nSolved at episode {i}: average reward: {running_reward:.2f}!")

    def evaluate(self) -> None:
        pass

    def _env_step(self, action: np.ndarray, other_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                               np.ndarray, np.ndarray]:
        state, reward, done, info = self.env.step(action, other_action)
        return (state.astype(np.float32), np.array(reward, dtype=np.int32),
                np.array(done, dtype=np.int32),
                np.array(info[Algorithm.OTHER_STATE], dtype=np.float32))

    def _tf_env_step(self, action: tf.Tensor,
                     other_action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self._env_step,
                                 [action, other_action],
                                 [tf.float32, tf.int32, tf.int32, tf.float32])

    def _get_action(self, action: np.ndarray) -> np.ndarray:
        return np.array([int(i) for i in bin(action)[2:].zfill(self.env.action_space.n)], dtype=np.int32) \
            if type(self.env.action_space) is gym.spaces.MultiBinary else np.array(action, dtype=np.int32)

    def _tf_get_action(self, action: tf.Tensor) -> tf.Tensor:
        return tf.numpy_function(self._get_action, [action], tf.int32)

    # TODO: Build model class for other model
    def _get_other_action(self, other_state: np.ndarray) -> np.ndarray:
        return np.array(self.other_model.predict(other_state), dtype=np.int32)

    def _tf_get_other_action(self, other_state: tf.Tensor) -> tf.Tensor:
        return tf.numpy_function(self._get_other_action, [other_state], tf.int32)

    def _run_episode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        action_prs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state
        other_state = initial_state

        for t in tf.range(self.max_steps):
            # Convert state into a batched ndarray (batch size = 1)
            state = tf.expand_dims(state, axis=0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = self.model(state)

            # Sample next action from the action probability distribution
            action = tf.squeeze(tf.random.categorical(
                action_logits_t, num_samples=1))
            action_prs_t = tf.nn.softmax(action_logits_t)

            # Store log probability of the action chosen
            action_prs = action_prs.write(t, action_prs_t[0, action])

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Get action and other action
            action = self._tf_get_action(action)
            other_action = self._tf_get_other_action(other_state)

            # Apply action to the environment to get next state and reward
            state, reward, done, other_state = self._tf_env_step(action, other_action)
            state.set_shape(initial_state_shape)
            other_state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            # TODO: Render
            # if self.render:
            #     self.env.render()
            #     time.sleep(Algorithm.RENDER_SLEEP_TIME)

            if tf.cast(done, tf.bool):
                break

        action_prs = action_prs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_prs, values, rewards

    def _get_expected_return(self, rewards: tf.Tensor, discount_rate: float, standardize: bool = True) -> tf.Tensor:
        num_rewards = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=num_rewards)

        rewards = tf.cast(rewards[::-1], tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape

        # Start from the end of `rewards` and accumulate reward sums into the
        # `returns` array
        for i in tf.range(num_rewards):
            reward = rewards[i]
            discounted_sum = reward + discount_rate * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + self.eps))

        return returns

    def _compute_loss(self, action_prs: tf.Tensor, values: tf.Tensor,
                      returns: tf.Tensor) -> tf.Tensor:
        advantage = returns - values

        action_log_prs = tf.math.log(action_prs)
        actor_loss = -tf.math.reduce_sum(action_log_prs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    @ tf.function
    def _train_step(self, initial_state: tf.Tensor, discount_rate: float) -> tf.Tensor:
        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_prs, values, rewards = self._run_episode(initial_state)

            # Calculate expected returns
            returns = self._get_expected_return(rewards, discount_rate)

            # Convert training data to appropriate shape
            action_prs, values, returns = [tf.expand_dims(i, axis=1) for i in [action_prs, values, returns]]

            # Calculating loss values to update our network
            loss = self._compute_loss(action_prs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward
