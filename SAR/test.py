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

PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'DQN')
load_path = os.path.join(PATH, 'Saved_data')
if not os.path.exists(load_path): os.makedirs(load_path) 
load_checkpoint_path = os.path.join(PATH, "12-06-2023 13h58m08s")
save_path = load_checkpoint_path
models_path = os.path.join(save_path, 'models')

print(os.listdir(models_path)[-1][0])