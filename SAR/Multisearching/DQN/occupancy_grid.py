from Grassfire import Grassfire
import numpy as np

def random_grid(self, rows=16, cols=16, obstacleProb=0.3):
    '''Return a 2D numpy array representing a grid of randomly placed
    obstacles (where the likelihood of any cell being an obstacle
    is given by obstacleProb) and randomized start/destination cells.
    '''
    obstacleGrid = np.random.random_sample((rows, cols))
    grid = Grassfire.UNVIS * np.ones((rows, cols), dtype=np.int)
    grid[obstacleGrid <= obstacleProb] = self.OBST

    # Randomly set start and destination cells.
    self.set_start_dest(grid)
    return grid