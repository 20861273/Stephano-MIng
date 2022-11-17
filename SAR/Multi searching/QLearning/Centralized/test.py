from enum import Enum
from collections import namedtuple
import numpy as np
import random
import math
import time
import keyboard
from cen_m_qlearning import QLearning
from cen_m_environment import Environment, Point
# [a for a in q_tables[i, state[0]] if a == 0]

env = Environment(2)
ql = QLearning(env)
a = []
state = np.empty((2,))

for i in range(env.grid.shape[0]):
    for j in range(env.grid.shape[1]):
        for k in range(env.grid.shape[0]):
            for l in range(env.grid.shape[1]):
                ql.pos[0] = Point(i,j)
                ql.pos[1] = Point(k,l)

                state[0] = int(ql.pos[0].y*env.grid.shape[1] + ql.pos[0].x)
                state[1] = int(ql.pos[1].y*env.grid.shape[1] + ql.pos[1].x)

                s = [int(state[0]*env.grid.shape[0]*env.grid.shape[1] + state[1]),
                    int(state[1]*env.grid.shape[0]*env.grid.shape[1] + state[0])]

                print(ql.pos[0], ql.pos[1], state, s)