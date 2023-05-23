import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.pyplot import cm
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from datetime import datetime

from collections import namedtuple
r = [[[1,2,13,14],[3,4,15,16],[5,6,17,18]], [[7,8,19,21],[9,0,20,22],[11,12,23,24]]]

# Get the length of the nested lists
m = len(r)
k = len(r[0])
n = len(r[0][0])

grouped_lists = [[list(sub) for sub in zip(*sublist)] for sublist in r]
ts_rewards = []
for r_i in range(4):
    for n_ts_rewards in grouped_lists:
        ts_rewards.append(n_ts_rewards[r_i])
    print(ts_rewards)

print(grouped_lists)