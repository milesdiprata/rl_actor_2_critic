from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

import gym
import slimevolleygym
import numpy as np

import a2c.arguments as arguments

PATHS = {
    arguments.OtherPolicy.SLIMEVOLLEY_BASELINE: None,
    arguments.OtherPolicy.SLIMEVOLLEY_CMA:
        "zoo/cmaes/slimevolley.cma.64.96.best.json",
    arguments.OtherPolicy.SLIMEVOLLEY_GA: "zoo/ga_sp/ga.json",
    arguments.OtherPolicy.STABLE_BASELINES_PPO: "zoo/ppo/best_model.zip",
    arguments.OtherPolicy.RANDOM: None,
}


class OtherPolicy(ABC):
    def __init__(self, path: str) -> None:
        self.model = self._load(path)

    @staticmethod
    @abstractmethod
    def _load(path: str) -> Any:
        pass

    def predict(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict(state)


class SlimevolleyBaselinePolicy(OtherPolicy):
    @ staticmethod
    def _load(path: str) -> slimevolleygym.BaselinePolicy:
        return slimevolleygym.BaselinePolicy()


class SlimevolleyCMAPolicy(OtherPolicy):
    @ staticmethod
    def _load(path: str) -> Any:
        raise NotImplementedError("Missing module slimevolley.mlp!")


class SlimevolleyGAPolicy(OtherPolicy):
    @ staticmethod
    def _load(path: str) -> Any:
        raise NotImplementedError("Missing module slimevolley.mlp!")


class StableBaselinesPPOPolicy(OtherPolicy):
    @ staticmethod
    def _load(path: str) -> Any:
        raise NotImplementedError(
            "PPO model best_model.zip not compatible with stable-baselines3!")

    def predict(self, state: np.ndarray) -> np.ndarray:
        action, _state = self.model.predict(state, deterministic=True)
        return action


class RandomPolicy(OtherPolicy):
    def __init__(self, action_space: Union[gym.spaces.Discrete, gym.spaces.MultiBinary]) -> None:
        self.action_space = action_space
        assert action_space, "Action space is None!"

    def predict(self, states: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray]]) -> np.ndarray:
        return self.action_space.sample()


def get(other_policy: arguments.OtherPolicy,
        action_space: Union[gym.spaces.Discrete, gym.spaces.MultiBinary] = None) -> OtherPolicy:
    if other_policy == arguments.OtherPolicy.SLIMEVOLLEY_BASELINE:
        return SlimevolleyBaselinePolicy(PATHS[other_policy])
    elif other_policy == arguments.OtherPolicy.SLIMEVOLLEY_CMA:
        return SlimevolleyCMAPolicy(PATHS[other_policy])
    elif other_policy == arguments.OtherPolicy.SLIMEVOLLEY_GA:
        return SlimevolleyGAPolicy(PATHS[other_policy])
    elif other_policy == arguments.OtherPolicy.STABLE_BASELINES_PPO:
        return StableBaselinesPPOPolicy(PATHS[other_policy])
    elif other_policy == arguments.OtherPolicy.RANDOM:
        return RandomPolicy(action_space)
    else:
        raise ValueError("Unknown gym!")
