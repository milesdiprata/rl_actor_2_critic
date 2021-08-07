import os
from enum import Enum
from typing import List, Tuple

from prettytable import PrettyTable, ALL
import gym
import numpy as np


MAZE_H = 10  # height of the entire grid in cells
MAZE_W = 10  # width of the entire grid in cells


class Maze:
    class State:
        CLEAR = " "
        WALL = "W"
        PIT = "P"
        AGENT = "A"
        GOAL = "G"

    def __init__(self, width: int, height: int, agent: np.ndarray,
                 goal: np.ndarray, walls: List[np.ndarray] = [],
                 pits: List[np.ndarray] = []) -> None:
        self.width = width
        self.height = height
        self.agent_start = agent
        self.agent = agent
        self.goal = goal
        self.walls = walls
        self.pits = pits
        self.grid = self._make_grid()

    def at(self, xy: np.ndarray) -> State:
        return self.grid[xy[1], xy[0]]

    def reset(self) -> np.ndarray:
        self.agent = self.agent_start
        self.grid = self._make_grid()
        return self.agent

    def move_agent(self, agent: np.ndarray) -> None:
        self.grid[self.agent[1], self.agent[0]] = Maze.State.CLEAR
        self.grid[agent[1], agent[0]] = Maze.State.AGENT
        self.agent = agent

    def _make_grid(self) -> np.ndarray:
        grid = np.array([[Maze.State.CLEAR] * self.width] * self.height)
        grid[self.agent[1], self.agent[0]] = Maze.State.AGENT
        grid[self.goal[1], self.goal[0]] = Maze.State.GOAL
        for wall in self.walls:
            grid[wall[1], wall[0]] = Maze.State.WALL
        for pit in self.pits:
            grid[pit[1], pit[0]] = Maze.State.PIT
        return grid


class MazeworldEnv(gym.Env):
    class Action(Enum):
        UP = 0
        DOWN = 1
        RIGHT = 2
        LEFT = 3

    NUM_ACTIONS = 4

    AGENT_START = np.array([0, 0])
    GOAL = np.array([4, 5])

    WALLS = np.array([[2, 2], [3, 2], [4, 2], [5, 2],
                      [2, 3], [2, 4], [2, 5], [2, 6], [2, 7],
                      [2, 8], [3, 8], [4, 8], [5, 8]])
    PITS = np.array([[6, 3], [1, 4]])

    def __init__(self, width: int = MAZE_W, height: int = MAZE_H) -> None:
        self.maze = Maze(width, height, MazeworldEnv.AGENT_START,
                         MazeworldEnv.GOAL, MazeworldEnv.WALLS,
                         MazeworldEnv.PITS)
        self.action_space = gym.spaces.Discrete(MazeworldEnv.NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(
            np.array([0, 0]), np.array([width - 1, height - 1]))
        self.reward_range = (-10, 1)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        state = self.maze.agent
        next_state = state.copy()

        if action == MazeworldEnv.Action.UP.value and state[1] > 0:
            next_state[1] -= 1
        elif action == MazeworldEnv.Action.DOWN.value \
                and state[1] < self.maze.height - 1:
            next_state[1] += 1
        elif action == MazeworldEnv.Action.RIGHT.value \
                and state[0] < self.maze.width - 1:
            next_state[0] += 1
        elif action == MazeworldEnv.Action.LEFT.value and state[0] > 0:
            next_state[0] -= 1

        reward, done, reverse_action = self._get_reward(next_state)
        if reverse_action:
            next_state = state
        else:
            self.maze.move_agent(next_state)

        return next_state, reward, done

    def reset(self) -> np.ndarray:
        return self.maze.reset()

    def render(self) -> None:
        # os.system("cls" if os.name == "nt" else "clear")
        print("\n" * 150)
        table = PrettyTable()
        for row in self.maze.grid:
            table.add_row(row)
        print(table.get_string(header=False, hrules=ALL))

    def seed(self, seed=None) -> None:
        pass

    def _get_reward(self, next_state: np.ndarray) -> Tuple[float, bool]:
        reward = -0.1
        done = False
        reverse_action = False

        if self.maze.at(next_state) == Maze.State.GOAL:
            reward = 1.0
            done = True
        elif self.maze.at(next_state) == Maze.State.WALL:
            reward = -0.3
            reverse_action = True
        elif self.maze.at(next_state) == Maze.State.PIT:
            reward = -10
            done = True
        return reward, done, reverse_action


gym.register(id="Mazeworld-v0",
             entry_point="mazeworldgym.mazeworld:MazeworldEnv")
