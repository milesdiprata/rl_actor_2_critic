import os
import sys
import time
from enum import Enum
from typing import List, Tuple

from prettytable import PrettyTable
import gym
import numpy as np
from numpy.lib.index_tricks import nd_grid
import prettytable

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels per cell (width and height)
MAZE_H = 10  # height of the entire grid in cells
MAZE_W = 10  # width of the entire grid in cells
origin = np.array([UNIT / 2, UNIT / 2])


class MazeTk(tk.Tk, object):
    def __init__(self, agentXY, goalXY, walls=[], pits=[]):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.wallblocks = []
        self.pitblocks = []
        self.UNIT = 40   # pixels per cell (width and height)
        self.MAZE_H = 10  # height of the entire grid in cells
        self.MAZE_W = 10  # width of the entire grid in cells
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self.build_shape_maze(agentXY, goalXY, walls, pits)

    def build_shape_maze(self, agentXY, goalXY, walls, pits):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        for x, y in walls:
            self.add_wall(x, y)
        for x, y in pits:
            self.add_pit(x, y)
        self.add_goal(goalXY[0], goalXY[1])
        self.add_agent(agentXY[0], agentXY[1])
        self.canvas.pack()

    '''Add a solid wall block at coordinate for centre of bloc'''

    def add_wall(self, x, y):
        wall_center = origin + np.array([UNIT * x, UNIT*y])
        self.wallblocks.append(self.canvas.create_rectangle(
            wall_center[0] - 15, wall_center[1] - 15,
            wall_center[0] + 15, wall_center[1] + 15,
            fill='black'))

    '''Add a solid pit block at coordinate for centre of bloc'''

    def add_pit(self, x, y):
        pit_center = origin + np.array([UNIT * x, UNIT*y])
        self.pitblocks.append(self.canvas.create_rectangle(
            pit_center[0] - 15, pit_center[1] - 15,
            pit_center[0] + 15, pit_center[1] + 15,
            fill='blue'))

    '''Add a solid goal for goal at coordinate for centre of bloc'''

    def add_goal(self, x=4, y=4):
        goal_center = origin + np.array([UNIT * x, UNIT*y])

        self.goal = self.canvas.create_oval(
            goal_center[0] - 15, goal_center[1] - 15,
            goal_center[0] + 15, goal_center[1] + 15,
            fill='yellow')

    '''Add a solid wall red block for agent at coordinate for centre of bloc'''

    def add_agent(self, x=0, y=0):
        agent_center = origin + np.array([UNIT * x, UNIT*y])

        self.agent = self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')

    def reset(self, value=1, resetAgent=True):
        self.update()
        time.sleep(0.2)
        if(value == 0):
            return self.canvas.coords(self.agent)
        else:
            if(resetAgent):
                self.canvas.delete(self.agent)
                self.agent = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                                                          origin[0] + 15, origin[1] + 15,
                                                          fill='red')

            return self.canvas.coords(self.agent)

    '''computeReward - definition of reward function'''

    def computeReward(self, currstate, action, nextstate):
        reverse = False
        if nextstate == self.canvas.coords(self.goal):
            reward = 1
            done = True
            nextstate = 'terminal'
        elif nextstate in [self.canvas.coords(w) for w in self.wallblocks]:
            reward = -0.3
            done = False
            nextstate = currstate
            reverse = True
        elif nextstate in [self.canvas.coords(w) for w in self.pitblocks]:
            reward = -10
            done = True
            nextstate = 'terminal'
            reverse = False
        else:
            reward = -0.1
            done = False
        return reward, done, reverse

    '''step - definition of one-step dynamics function'''

    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.agent)  # next state

        # call the reward function
        reward, done, reverse = self.computeReward(s, action, s_)
        if(reverse):
            self.canvas.move(self.agent, -base_action[0], -base_action[1])  # move agent back
            s_ = self.canvas.coords(self.agent)

        return s_, reward, done

    def render(self, sim_speed=.01):
        time.sleep(sim_speed)
        self.update()


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
        os.system('cls' if os.name == 'nt' else 'clear')
        table = PrettyTable()
        for row in self.maze.grid:
            table.add_row(row)
        print(table.get_string(header=False, hrules=prettytable.ALL))

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
             entry_point='mazeworldgym.mazeworld:MazeworldEnv')
