#!/usr/bin/env python
import warnings

import gym
import numpy as np
import slimevolleygym
import tensorflow as tf

import a2c

GYM_ENV_NAMES = {
    a2c.arguments.Gym.SLIMEVOLLEY: "SlimeVolley-v0",
    a2c.arguments.Gym.MAZEWORLD: "mazeworld"
}

MODEL_PATH = "models/a2c/{}_model.tf"

warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="gym")

np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)


def slimevolley(args: a2c.arguments.Arguments) -> None:
    pass


def mazeworld() -> None:
    pass


def main() -> None:
    args = a2c.arguments.Arguments()
    # if args.gym == arguments.Gym.SLIMEVOLLEY:
    #     slimevolley(args)
    # elif args.gym == arguments.Gym.MAZEWORLD:
    #     mazeworld(args)
    # else:
    #     raise ValueError("Unknown gym!")

    model_path = MODEL_PATH.format(args.gym.name.lower())

    if args.day:
        slimevolleygym.setDayColors()

    env = gym.make(GYM_ENV_NAMES[args.gym])
    env.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    other_policy = a2c.other_policy.get(args.other_policy,
                                        action_space=env.action_space)

    algo = a2c.algorithm.Algorithm(env, other_policy,
                                   args.max_episodes, args.max_steps)
    if args.train:
        algo.train()
        algo.save_model(model_path)
    else:
        algo.load_model(model_path)
    for _i in range(args.max_episodes):
        cumulative_score = algo.render_episode()
        print("Cumulative Score:", cumulative_score)


if __name__ == "__main__":
    main()
