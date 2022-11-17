from enum import Enum
from collections import namedtuple
import numpy as np
import random
import math
import time
import keyboard
import pandas as pd


# initialize list of lists

states = []
pos = []
exp = 7
epoch = 2
for i in range(12):
    for j in range(12):
        pos.append(i,j)
        states.append([i*3 +j])
data = {'state': states}

reward = [[0]*exp]*len(states)

for i in range(epoch):
    data['epoch %d rewards' %(i)] = reward

  
# Create the pandas DataFrame
df = pd.DataFrame(data)

rewards = df['epoch 0 rewards']
  
# print dataframe.
for i in range(144):
    print(i, df.loc[i,'state'])