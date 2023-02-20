import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import os

class print_results:
    """
    A class used to print the results
    ...
    Attributes
    ----------
    grid : int
        3D array of grid-based environment at each time step. (grid[time_step, y, x])
    rows : int
        number of rows in the environment
    cols : int
        number of columns in the environment
    n_r : int
        the number of robots
    Methods
    -------
    def print_graph(self):
        prints the grid environment
    """

    def __init__(self,grid,rows,cols):
        self.grid = grid
        self.rows = rows
        self.cols = cols
    def print_graph(self, path):
        """
        Prints the grid environment
        """

        plt.rc('font', size=12)
        plt.rc('axes', titlesize=15) 

        # Prints graph
        fig,ax = plt.subplots(figsize=(8, 8))

        ax.set_aspect("equal")
        ax.set_xlim(0.5, self.cols + 0.5)
        ax.set_ylim(0.5, self.rows + 0.5)
        # Set tick positions to be centered between grid lines
        ax.set_xticks(np.arange(self.cols) + 0.5)
        ax.set_yticks(np.arange(self.rows) + 0.5)

        # Set tick labels to be the x or y coordinate of the grid cell
        ax.set_xticklabels(np.arange(self.cols))
        ax.set_yticklabels(np.arange(self.rows))

        # Adjust tick label position and font size
        ax.tick_params(axis='both', labelsize=10, pad=2, width=0.5, length=2)
        ax.grid(True, color='black', linewidth=1)

        for i in range(self.rows+1):
            for j in range(self.cols+1):
                ax.fill([j, j + 1, j + 1, j], [i, i, i + 1, i + 1], facecolor="white", alpha=0.5)

        # Add path to plot
        for i, point in enumerate(path):
            x, y = point.x, point.y
            label = str(i)
            if path.count(point) > 1:
                label = ", ".join([str(j) for j, p in enumerate(path) if p == point])
            ax.text(x+1, y+1, label, ha="center", va="center", color="black", fontsize=14)
            ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], [y + 0.5, y + 0.5, y + 1.5, y + 1.5], facecolor="green", alpha=0.5)
        
        plt.show()
        