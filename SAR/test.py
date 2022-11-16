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
exp = 7
epoch = 2
for i in range(12):
    for j in range(12):
        states.append([i,j])
data = {'state': states}

reward = [[0]*exp]*len(states)

for i in range(epoch):
    data['epoch %d rewards' %(i)] = reward

  
# Create the pandas DataFrame
df = pd.DataFrame(data)

rewards = df['epoch 0 rewards']
  
# print dataframe.
print(df, rewards[0][0])