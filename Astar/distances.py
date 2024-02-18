from gdistance.raster import Raster
from gdistance.gdistance import GDistance
# from gdistance.utils import *

import numpy as np

def mean(x1, x2):
    return np.divide(2, np.add(x1, x2))

ncols, nrows = 7,6
minX, minY = 0, 0
xres, yres = 1, 1
maxX = minX + (ncols * xres)
maxY = minY + (nrows * xres)
values = [[2, 2, 1, 1, 5, 5, 5], 
          [2, 2, 8, 8, 5, 2, 1], 
          [7, 1, 1, 8, 2, 2, 2], 
          [8, 7, 8, 8, 8, 8, 5], 
          [8, 8, 1, 1, 5, 3, 9], 
          [8, 1, 1, 2, 5, 3, 9]]

raster = Raster(extent=[minX, maxX,minY, maxY], xres=xres, yres=yres, crs=3857, nodatavalue=-9999, pix_values=values)

gd = GDistance()
trans = gd.transition(raster, function=mean, directions=4)
targets = [(5.5, 1.5)]
accost = gd.acc_cost(trans, targets)

print(accost)