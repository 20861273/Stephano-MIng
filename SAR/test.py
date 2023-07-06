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

collision = {  'obstacle' :   [False]*2,
                'boundary' :   [True]*2,
                'drone'    :   [False]*2}

if any(any(collision_tpye) for collision_tpye in collision.values()): print("hi")