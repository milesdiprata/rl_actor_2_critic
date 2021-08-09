# Actor 2 Critic Algorithm
Actor 2 Critic reinforcement learning algorithm implementation using TensorFlow.

## Requirements
* [Anaconda](https://docs.anaconda.com/anaconda/install/) 

## Installation
For Mac OSX:
```
make requirements_osx
make env
conda activate rl_actor_2_critic
```

For Ubuntu:
```
make requirements_ubuntu
make env
conda activate rl_actor_2_critic
```

## Running the Program
Display program usage:
```
# python3 a2c.py -h
usage: a2c.py [-h] [--gym GYM] [--train] [--other OTHER] [--render] [--day] [--seed SEED] [--episodes EPISODES]
              [--steps STEPS]

Evaluate A2C agent against an opponent (Option 2).

optional arguments:
  -h, --help           show this help message and exit
  --gym GYM            Choice of gym environment: slimevolley or mazeworld.
  --train              Train and save new model?
  --other OTHER        Choice of other policy (for slimevolley): baseline, ppo, cma, ga or random.
  --day                Daytime colors?
  --seed SEED          Random seed (integer); default 721.
  --episodes EPISODES  Number of episodes (integer); default 10000.
  --steps STEPS        Max. number of steps per episode (integer); default 1000.
```

*e.g.*: Training in `slimevolley` environment for `T = 1000` episodes
```
python3 --gym slimevolley --train --episodes 10000
```

*e.g.*: Evaluating existing model in `slimevolley` environment for `N = 100` episodes and `T = 100` steps per episodeP
```
python3 --gym slimevolley --episodes 100 --steps 100
```
