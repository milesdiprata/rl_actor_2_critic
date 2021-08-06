import argparse
from enum import Enum


class Gym(Enum):
    SLIMEVOLLEY = 1
    MAZEWORLD = 2


class OtherPolicy(Enum):
    SLIMEVOLLEY_BASELINE = 1
    SLIMEVOLLEY_CMA = 2
    SLIMEVOLLEY_GA = 3
    STABLE_BASELINES_PPO = 4
    RANDOM = 5


class Arguments:
    GYMS = {
        "slimevolley": Gym.SLIMEVOLLEY,
        "mazeworld": Gym.MAZEWORLD
    }

    OTHER_POLICIES = {
        "baseline": OtherPolicy.SLIMEVOLLEY_BASELINE,
        "cma": OtherPolicy.SLIMEVOLLEY_CMA,
        "ga": OtherPolicy.SLIMEVOLLEY_GA,
        "ppo": OtherPolicy.STABLE_BASELINES_PPO,
        "random": OtherPolicy.RANDOM
    }

    def __init__(self) -> None:
        args = Arguments._parse_args()
        self.gym = Arguments.GYMS[args.gym]
        self.train = bool(args.train)
        self.other_policy = Arguments.OTHER_POLICIES[args.other]
        self.render = bool(args.render)
        self.day = bool(args.day)
        self.seed = int(args.seed)
        self.max_episodes = int(args.episodes)
        self.max_steps = int(args.steps)

    @staticmethod
    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Evaluate A2C agent against an opponent (Option 2).")
        parser.add_argument("--gym", help="Choice of gym environment: slimevolley or mazeworld.",
                            type=str, default="slimevolley")
        parser.add_argument("--train", action="store_true", help="Train and save new model?", default=False)
        parser.add_argument("--other", help="Choice of other policy (for slimevolley): baseline, ppo, cma, ga or random.",
                            type=str, default="baseline")
        parser.add_argument("--render", action="store_true", help="Render to screen?", default=False)
        parser.add_argument("--day", action="store_true", help="Daytime colors?", default=False)
        parser.add_argument("--seed", help="Random seed (integer); default 721.", type=int, default=721)
        parser.add_argument("--episodes", help="Number of episodes (integer); default 10000).",
                            type=int, default=10000)
        parser.add_argument(
            "--steps", help="Max. number of steps per episode (integer); default 1000.", type=int, default=1000)

        return parser.parse_args()
