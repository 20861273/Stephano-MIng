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

def write_json(lst, file_name):
    with open(file_name, "w") as f:
        json.dump(lst, f)

PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'DQN')
load_path = os.path.join(PATH, 'Saved_data')
if not os.path.exists(load_path): os.makedirs(load_path)
date_and_time = datetime.now()
save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
if not os.path.exists(save_path): os.makedirs(save_path)
models_path = os.path.join(save_path, 'models')
if not os.path.exists(models_path): os.makedirs(models_path)

hp = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" %(\
        str(1), \
        str(0.0001),\
        str(0.9), \
        str(0.01), \
        str(1), \
        str(0.1), \
        str(0), \
        str(0.1), \
        str(200), \
        "image",\
        "goal", \
        "random", \
        str(1000))
file_name = "hyperparameters%s.json" %(str(0))
file_name = os.path.join(save_path, file_name)
write_json(hp, file_name)
