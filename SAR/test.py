import math
import numpy as np
import os

PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'QLearning')
PATH = os.path.join(PATH, '31-10-2022 16h33m22s')

file_name = "policy_rewards0.txt"
p=np.loadtxt(os.path.join(PATH, file_name))

np.savetxt(os.path.join(PATH, file_name), p[::500])


