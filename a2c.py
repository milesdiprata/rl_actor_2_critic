#!/usr/bin/env python
import argparse
import warnings

import gym
import numpy as np
import slimevolleygym
import tensorflow as tf

import a2c.eval
import a2c.model

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
    MODEL_BASELINE: a2c.model.make_baseline_policy,
    # MODEL_PPO: a2c.model.PPO,
    MODEL_CMA: slimevolleygym.mlp.makeSlimePolicy,
    MODEL_GA: slimevolleygym.mlp.makeSlimePolicyLite,
    MODEL_RANDOM: a2c.model.RandomPolicy,
}

warnings.filterwarnings("ignore", module='tensorflow')
warnings.filterwarnings("ignore", module='gym')

np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate A2C agent against an opponent (Option 2).")
    parser.add_argument(
        "--other", help="Choice of other agent (baseline, ppo, cma, ga, random).",
        type=str, default="baseline")
    parser.add_argument("--render", action="store_true",
                        help="Render to screen?", default=False)
    parser.add_argument("--day", action="store_true",
                        help="Daytime colors?", default=False)
    parser.add_argument(
        "--seed", help="Random seed (integer).", type=int, default=721)
    parser.add_argument(
        "--trials", help="Number of trials (default 1000).", type=int,
        default=1000)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.day:
        slimevolleygym.setDayColors()
    assert args.other in MODEL_PATHS.keys(), "Other agent model is not valid!"

    env = gym.make(ENV_NAME)
    env.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    eval = a2c.eval.Evaluate(
        env, a2c.model.A2C(
            2 ** env.action_space.n
            if type(env.action_space) is gym.spaces.MultiBinary
            else env.action_space.n),
        MODELS[args.other](MODEL_PATHS[args.other]),
        args.trials, render=args.render)

    eval.test()


if __name__ == "__main__":
    main()
