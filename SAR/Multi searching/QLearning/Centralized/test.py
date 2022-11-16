from enum import Enum
from collections import namedtuple
import numpy as np
import random
import math
import time
import keyboard
from cen_m_qlearning import QLearning
from cen_m_environment import Environment, Point

env = Environment(2)
ql = QLearning(env)
a = []

for i in range(env.grid.shape[0]):
    for j in range(env.grid.shape[1]):
        for k in range(env.grid.shape[0]):
            for l in range(env.grid.shape[1]):
                ql.pos[0] = Point(i,j)
                ql.pos[1] = Point(k,l)
                print(ql.pos[0], ql.pos[1], ql.get_state(env))