#!/usr/bin/env python
from tensorflow.python.types.core import Value
from a2c import arguments
import warnings

import gym
import numpy as np
import slimevolleygym
import mazeworldgym
import tensorflow as tf

import a2c

GYM_ENV_NAMES = {
    a2c.arguments.Gym.SLIMEVOLLEY: "SlimeVolley-v0",
    a2c.arguments.Gym.MAZEWORLD: "Mazeworld-v0"
}

MODEL_PATH = "models/a2c/{}_model.tf"

warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="gym")

# np.set_printoptions(threshold=20, precision=4, suppress=True, linewidth=200)
np.set_printoptions(threshold=np.inf)


def main() -> None:

    args = a2c.arguments.Arguments()
    model_path = MODEL_PATH.format(args.gym.name.lower())

    env = gym.make(GYM_ENV_NAMES[args.gym])
    env.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    algo = None
    if args.gym == a2c.arguments.Gym.SLIMEVOLLEY:
        if args.day:
            slimevolleygym.setDayColors()
        other_policy = a2c.other_policy.get(args.other_policy,
                                            action_space=env.action_space)
        algo = a2c.algorithm.Slimevolley(env, args.max_episodes,
                                         args.max_steps, other_policy)
    elif args.gym == a2c.arguments.Gym.MAZEWORLD:
        algo = a2c.algorithm.Mazeworld(env, args.max_episodes,
                                       args.max_steps)
    else:
        raise ValueError("Unknown gym!")

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
