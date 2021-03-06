import collections
import csv
import statistics
import time
from abc import ABC, abstractmethod
from typing import List, Tuple

import gym
import numpy as np
import tensorflow as tf
import tqdm

from a2c.other_policy import OtherPolicy


class Model(tf.keras.Model):
    def __init__(self, num_actions: int, num_hidden_units: int = 128) -> None:
        super().__init__()

        # TODO: Hidden layer for values?
        self.common_hidden = tf.keras.layers.Dense(num_hidden_units,
                                                   activation="relu")
        self.logits = tf.keras.layers.Dense(num_actions)
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        common_out = self.common_hidden(inputs)
        return self.logits(common_out), self.value(common_out)


class Algorithm(ABC):
    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.99

    MIN_EPISODE_CRITERION = 100

    def __init__(self, env: gym.Env, model: tf.keras.Model,
                 max_episodes: int, max_steps: int) -> None:
        self.env = env
        self.model = model
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.huber_loss = tf.keras.losses.Huber(
            reduction=tf.keras.losses.Reduction.SUM)
        self.eps = np.finfo(np.float32).eps.item()
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=Algorithm.LEARNING_RATE)

    def load_model(self, file_path: str) -> None:
        self.model = tf.keras.models.load_model(file_path)

    def save_model(self, file_path: str) -> None:
        self.model.save(file_path)

    def evaluate(self) -> None:
        for _i in range(self.max_episodes):
            cumulative_score = self.evaluate_step()
            print("Cumulative Score:", cumulative_score)

    def train(self, results_csv_name: str) -> None:
        with open(results_csv_name, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ["Episode", "Episode Reward", "Running Reward"])

            running_reward = 0

            # Keep last episodes reward
            episodes_reward = collections.deque(
                maxlen=Algorithm.MIN_EPISODE_CRITERION)

            with tqdm.trange(self.max_episodes) as t:
                for i in t:
                    initial_state = tf.constant(self.env.reset(),
                                                dtype=tf.float32)
                    episode_reward = int(self._train_step(
                        initial_state, Algorithm.DISCOUNT_FACTOR))

                    episodes_reward.append(episode_reward)
                    running_reward = statistics.mean(episodes_reward)

                    t.set_description(f"Episode {i}")
                    t.set_postfix(episode_reward=episode_reward,
                                  running_reward=running_reward)

                    csv_writer.writerow([i, episode_reward, running_reward])

            csv_file.close()

    @abstractmethod
    def render_episode(self) -> float:
        pass

    @abstractmethod
    def _env_step(self, action: np.ndarray,
                  other_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _tf_env_step(self, action: tf.Tensor,
                     other_action: tf.Tensor) -> List[tf.Tensor]:
        pass

    @abstractmethod
    def _run_episode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor,
                                                              tf.Tensor,
                                                              tf.Tensor]:
        pass

    def _get_expected_return(self, rewards: tf.Tensor, discount_rate: float,
                             standardize: bool = True) -> tf.Tensor:
        num_rewards = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=num_rewards)

        # Start from the end of `rewards` and accumulate reward sums into the
        # `returns` array
        rewards = tf.cast(rewards[::-1], tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape

        for i in tf.range(num_rewards):
            reward = rewards[i]
            discounted_sum = reward + discount_rate * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns))
                       / (tf.math.reduce_std(returns) + self.eps))

        return returns

    def _compute_loss(self, action_prs: tf.Tensor, values: tf.Tensor,
                      returns: tf.Tensor) -> tf.Tensor:
        advantage = returns - values

        action_log_prs = tf.math.log(action_prs)
        actor_loss = -tf.math.reduce_sum(action_log_prs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    @ tf.function
    def _train_step(self, initial_state: tf.Tensor,
                    discount_rate: float) -> tf.Tensor:
        with tf.GradientTape() as tape:
            # Run the model for one episode to collect training data
            action_prs, values, rewards = self._run_episode(initial_state)

            # Calculate expected returns
            returns = self._get_expected_return(rewards, discount_rate)

            # Convert training data to appropriate shape
            action_prs, values, returns = [tf.expand_dims(i, axis=1) for i in
                                           [action_prs, values, returns]]

            # Calculating loss values to update our network
            loss = self._compute_loss(action_prs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads,
                                           self.model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward


class Slimevolley(Algorithm):
    RENDER_SLEEP_TIME = 0.2
    OTHER_STATE = "otherObs"

    def __init__(self, env: gym.Env, max_episodes: int,
                 max_steps: int, other_policy: OtherPolicy) -> None:
        super().__init__(env, Model(2 ** env.action_space.n),
                         max_episodes, max_steps)
        self.other_policy = other_policy

    def render_episode(self) -> float:
        state = tf.constant(self.env.reset(), dtype=tf.float32)
        other_state = state if self.other_policy is not None else None

        total_reward = 0

        for _i in range(self.max_steps):
            state = tf.expand_dims(state, axis=0)

            action_logits, _value = self.model(state)
            action = tf.argmax(tf.squeeze(action_logits))

            action = self._tf_get_action(action)
            other_action = self._tf_get_other_action(other_state)

            state, reward, done, other_state = self._tf_env_step(action,
                                                                 other_action)
            total_reward += reward

            self.env.render()
            time.sleep(Slimevolley.RENDER_SLEEP_TIME)

            if tf.cast(done, tf.bool):
                break

        return total_reward

    def _env_step(self, action: np.ndarray,
                  other_action: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
        state, reward, done, info = self.env.step(action, other_action)
        return (state.astype(np.float32),
                np.array(reward, dtype=np.int32),
                np.array(done, dtype=np.int32),
                np.array(info[Slimevolley.OTHER_STATE], dtype=np.float32))

    def _tf_env_step(self, action: tf.Tensor,
                     other_action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self._env_step,
                                 [action, other_action],
                                 [tf.float32, tf.int32, tf.int32, tf.float32])

    def _get_action(self, action: np.ndarray) -> np.ndarray:
        return np.array([int(i) for i in bin(action)[2:].zfill(
            self.env.action_space.n)], dtype=np.int32)

    def _tf_get_action(self, action: tf.Tensor) -> tf.Tensor:
        return tf.numpy_function(self._get_action, [action], tf.int32)

    # TODO: Build model class for other model
    def _get_other_action(self, other_state: np.ndarray) -> np.ndarray:
        return np.array(self.other_policy.predict(other_state), dtype=np.int32)

    def _tf_get_other_action(self, other_state: tf.Tensor) -> tf.Tensor:
        return tf.numpy_function(self._get_other_action,
                                 [other_state], tf.int32)

    def _run_episode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor,
                                                              tf.Tensor,
                                                              tf.Tensor]:
        action_prs = tf.TensorArray(dtype=tf.float32, size=0,
                                    dynamic_size=True)
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
            action = tf.squeeze(tf.random.categorical(action_logits_t,
                                                      num_samples=1))
            action_prs_t = tf.nn.softmax(action_logits_t)

            # Store log probability of the action chosen
            action_prs = action_prs.write(t, action_prs_t[0, action])

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Get action and other action
            action = self._tf_get_action(action)
            other_action = self._tf_get_other_action(other_state)

            # Apply action to the environment to get next state and reward
            state, reward, done, other_state = self._tf_env_step(action,
                                                                 other_action)
            state.set_shape(initial_state_shape)
            other_state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_prs = action_prs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_prs, values, rewards


class Mazeworld(Algorithm):
    RENDER_SLEEP_TIME = 0.2

    def __init__(self, env: gym.Env,
                 max_episodes: int, max_steps: int) -> None:
        super().__init__(env, Model(env.action_space.n),
                         max_episodes, max_steps)

    def render_episode(self) -> float:
        state = tf.constant(self.env.reset(), dtype=tf.int32)
        total_reward = 0

        for _i in range(self.max_steps):
            state = tf.expand_dims(state, axis=0)

            action_logits, _value = self.model(state)
            action = tf.argmax(tf.squeeze(action_logits))

            state, reward, done = self._tf_env_step(action)
            total_reward += reward

            self.env.render()
            time.sleep(Mazeworld.RENDER_SLEEP_TIME)

            if done:
                break

        return total_reward

    def _env_step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
        state, reward, done = self.env.step(action)
        return (state.astype(np.int32),
                np.array(reward, dtype=np.float32),
                np.array(done, dtype=np.int32))

    def _tf_env_step(self, action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self._env_step,
                                 [action],
                                 [tf.int32, tf.float32, tf.int32])

    def _run_episode(self, initial_state: tf.Tensor) -> Tuple[tf.Tensor,
                                                              tf.Tensor,
                                                              tf.Tensor]:
        action_prs = tf.TensorArray(dtype=tf.float32, size=0,
                                    dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = tf.cast(initial_state, tf.int32)

        for t in tf.range(self.max_steps):
            # Convert state into a batched ndarray (batch size = 1)
            state = tf.expand_dims(state, axis=0)

            # Run the model and to get action probabilities and critic value
            action_logits_t, value = self.model(state)

            # Sample next action from the action probability distribution
            action = tf.squeeze(tf.random.categorical(action_logits_t,
                                                      num_samples=1))
            action_prs_t = tf.nn.softmax(action_logits_t)

            # Store log probability of the action chosen
            action_prs = action_prs.write(t, action_prs_t[0, action])

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Apply action to the environment to get next state and reward
            state, reward, done = self._tf_env_step(action)
            state.set_shape(initial_state_shape)

            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                break

        action_prs = action_prs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_prs, values, rewards
