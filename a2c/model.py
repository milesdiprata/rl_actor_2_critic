from typing import Tuple

import gym
import slimevolleygym
import stable_baselines3
import tensorflow as tf


class A2C(tf.keras.Model):
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


# class PPO:
#     def __init__(self, path):
#         self.model = stable_baselines.PPO1.load(path)

#     def predict(self, obs):
#         action, _state = self.model.predict(obs, deterministic=True)
#         return action


class RandomPolicy:
    def __init__(self, _path):
        self.action_space = gym.spaces.MultiBinary(3)

    def predict(self, _obs):
        return self.action_space.sample()


def make_baseline_policy(_path):
    return slimevolleygym.BaselinePolicy()
