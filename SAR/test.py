import math
import numpy as np
import os

asb_path = os.getcwd()
asb_path = os.path.join(asb_path, 'SAR')
asb_path = os.path.join(asb_path, 'Results')
asb_path = os.path.join(asb_path, 'QLearning')

walk = list(os.walk(asb_path))
for path, _, _ in walk[::1]:
    if len(os.listdir(path)) == 0:
        os.rmdir(path)
