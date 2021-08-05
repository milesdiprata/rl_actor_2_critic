#!/usr/bin/env python
import argparse
from re import A
import warnings

import gym
import numpy as np
import slimevolleygym
import tensorflow as tf

import a2c.algorithm as algorithm
import a2c.model as model

ENV_NAME = "SlimeVolley-v0"

MODEL_BASELINE = "baseline"
MODEL_PPO = "ppo"
MODEL_GA = "ga"
MODEL_CMA = "cma"
MODEL_RANDOM = "random"

MODEL_PATHS = {
    MODEL_BASELINE: None,
    MODEL_PPO: "zoo/ppo/best_model.zip",
    MODEL_CMA: "zoo/cmaes/slimevolley.cma.64.96.best.json",
    MODEL_GA: "zoo/ga_sp/ga.json",
    MODEL_RANDOM: None,
}

MODELS = {
    MODEL_BASELINE: model.make_baseline_policy,
    # MODEL_PPO: model.PPO,
    MODEL_CMA: slimevolleygym.mlp.makeSlimePolicy,
    MODEL_GA: slimevolleygym.mlp.makeSlimePolicyLite,
    MODEL_RANDOM: model.RandomPolicy,
}

A2C_MODEL_FILE_PATH = "models/model.zip"

warnings.filterwarnings("ignore", module='tensorflow')
warnings.filterwarnings("ignore", module='gym')

np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate A2C agent against an opponent (Option 2).")
    parser.add_argument("--train", action="store_true", help="Load saved model?", default=False)
    parser.add_argument("--other", help="Choice of other agent (baseline, ppo, cma, ga, random).",
                        type=str, default="baseline")
    parser.add_argument("--render", action="store_true", help="Render to screen?", default=False)
    parser.add_argument("--day", action="store_true", help="Daytime colors?", default=False)
    parser.add_argument("--seed", help="Random seed (integer).", type=int, default=721)
    parser.add_argument("--episodes", help="Number of episodes (default 10000).",
                        type=int, default=10000)
    parser.add_argument("--steps", help="Number of training steps (default 1000).", type=int, default=1000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.day:
        slimevolleygym.setDayColors()
    assert args.other in MODEL_PATHS.keys(), "Other agent model is not valid!"

    env = gym.make(ENV_NAME)
    env.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    other_model = MODELS[args.other](MODEL_PATHS[args.other])

    algo = algorithm.Algorithm(env, other_model, args.episodes, args.steps)
    if args.train:
        algo.train()
        algo.save_model(A2C_MODEL_FILE_PATH)
    else:
        algo.load_model(A2C_MODEL_FILE_PATH)
    algo.evaluate()


if __name__ == "__main__":
    main()
