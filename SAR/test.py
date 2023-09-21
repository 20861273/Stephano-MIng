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
Point = namedtuple('Point', 'x, y')

w1 = np.arange(-1, 10, 0.1)
w2 = np.arange(-1, 10, 0.1)
a = np.arange(-1, 10, 0.1)
b = np.arange(-1, 10, 0.1)

c = [8,8,3,2]
d = [7,3,4,1]

store = []
f = np.array([0,0,0,0])

for i in w1:
    for j in w2:
        for k in a:
            for l in b:
                for m in range(len(c)):
                    f[m] = i * math.exp(-k*c[m]) + j * math.exp(-l*d[m])
                if f[3] < f[0] and f[0]<f[1] and f[1]<f[2]:
                    store.append[(w1,w2,a,b)]


store


# def gradual_increase_ones(array, current_time_step):
#     # Determine the dimensions of the input 2D array
#     height, width = array.shape

#     # Calculate the current percentage of zeros in the array
#     current_zeros_percentage = 1 - np.count_nonzero(array) / (height * width)

#     if current_time_step <= 30:  # Gradually increase center block to 10% of the array
#         # Calculate the target number of zeros for the center block
#         target_zeros_count = int(0.1 * height * width)

#         # Calculate the side length of the center block
#         block_side_length = int(np.sqrt(target_zeros_count))

#         # Calculate the starting position of the center block
#         start_pos = (height - block_side_length) // 2

#         # Fill the center block with zeros
#         array[start_pos:start_pos+block_side_length, start_pos:start_pos+block_side_length] = 0

#     elif current_time_step <= 60:  # Gradually increase center block to 40% of the array
#         # Calculate the target number of zeros for the center block
#         target_zeros_count = int(0.3 * height * width)

#         # Calculate the side length of the center block
#         block_side_length = int(np.sqrt(target_zeros_count))

#         # Calculate the starting position of the center block
#         start_pos = (height - block_side_length) // 2

#         # Fill the center block with zeros
#         array[start_pos:start_pos+block_side_length, start_pos:start_pos+block_side_length] = 0

#     elif current_time_step <= 90:  # Gradually increase center block to 70% of the array
#         # Calculate the target number of zeros for the center block
#         target_zeros_count = int(0.6 * height * width)

#         # Calculate the side length of the center block
#         block_side_length = int(np.sqrt(target_zeros_count))

#         # Calculate the starting position of the center block
#         start_pos = (height - block_side_length) // 2

#         # Fill the center block with zeros
#         array[start_pos:start_pos+block_side_length, start_pos:start_pos+block_side_length] = 0

#     else:  # Fill only the border with ones, rest zeros
#         array[1:-1, 1:-1] = 0
#         array[0, :] = 1
#         array[-1, :] = 1
#         array[:, 0] = 1
#         array[:, -1] = 1

#     return array

# array = np.ones((20,20))
# first = False

# for i in range(100):
#     array = gradual_increase_ones(array, i)
#     if first:
#         print(array)
#         first = False


# l = [[1,2,3],[4,5,6],[7,8,9]]

# l.append([1])

# print(l)