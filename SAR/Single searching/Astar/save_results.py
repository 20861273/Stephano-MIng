import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import os

from astar_environment import Environment, HEIGHT, WIDTH, Point

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
    def print_graph(self, paths, cost, time, dir_path, pos_cnt):
        """
        Prints the grid environment
        """

        plt.rc('font', size=20)
        plt.rc('axes', titlesize=10)

        visited = []
        cnt = 0

        for path in paths:
            # Prints graph
            fig,ax = plt.subplots(figsize=(WIDTH, HEIGHT))

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

            for i in range(HEIGHT): # y
                for j in range(WIDTH): # x
                    ax.fill([j, j + 1, j + 1, j], [i, i, i + 1, i + 1], facecolor="white", alpha=0.5)

            # Add path to plot
            for i in visited:
                x, y = i.x, i.y
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5],
                    [y + 0.5, y + 0.5, y + 1.5, y + 1.5],
                    facecolor="gray",
                    alpha=0.5)

            clabel = ""
            grid = np.zeros((HEIGHT, WIDTH))
            row_visit = []
            for i, point in enumerate(path):
                if i == len(path)-1: break
                x, y = point.x, point.y
                label = []
                for j, p in enumerate(path):
                    if p == point and p not in row_visit:
                        if j == len(path)-1: break
                        if path[j+1].x == x+1:
                            clabel = ">"
                            grid[p.y][p.x] += 1
                        elif path[j+1].x == x-1:
                            clabel = "<"
                            grid[p.y][p.x] += 1
                        elif path[j+1].y == y+1:
                            clabel = "^"
                            grid[p.y][p.x] += 1
                        elif path[j+1].y == y-1:
                            clabel = "v"
                            grid[p.y][p.x] += 1
                        elif path[j+1].x == x and path[j+1].y == y:
                            clabel = "-"
                            grid[p.y][p.x] += 1

                        if grid[p.y, p.x] % 3 == 0 and grid[p.y, p.x] != 0:
                            temp = clabel + "\n"
                            label.append(temp)
                        else:
                            label.append(clabel)

                label = " | ".join(label)
                visited.append(point)
                row_visit.append(point)
                
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                        [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                        facecolor="green", 
                        alpha=0.5)
                ax.text(x+1, y+1, label, ha="center", va="center", color="black", fontsize=8)
                
            
            plt_title = "Adapted A* algorithm:\nCost: %s\nTime: %s" %(str(cost), str(time)) 
            plt.title(plt_title)

            file_name = "traj%s_%s.png"%(str(pos_cnt), str(cnt))
            plt.savefig(os.path.join(dir_path, file_name))
            cnt += 1
        plt.close()

    def print_row(self, path, dir_path, id, visited):
        """
        Prints the grid environment
        """

        plt.rc('font', size=20)
        plt.rc('axes', titlesize=10)

        # Prints graph
        fig,ax = plt.subplots(figsize=(WIDTH, HEIGHT))

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
        for row in visited:
            for i in row:
                x, y = i.x, i.y
                ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5],
                        [y + 0.5, y + 0.5, y + 1.5, y + 1.5],
                        facecolor="gray",
                        alpha=0.5)

        clabel = ""
        grid = np.zeros((HEIGHT, WIDTH))
        row_visit = []
        for i, point in enumerate(path):
            if i == len(path)-1: break
            x, y = point.x, point.y
            label = []
            for j, p in enumerate(path):
                if p == point and p not in row_visit:
                    if j == len(path)-1: break
                    if path[j+1].x == x+1:
                        clabel = ">"
                        grid[p.y][p.x] += 1
                    elif path[j+1].x == x-1:
                        clabel = "<"
                        grid[p.y][p.x] += 1
                    elif path[j+1].y == y+1:
                        clabel = "^"
                        grid[p.y][p.x] += 1
                    elif path[j+1].y == y-1:
                        clabel = "v"
                        grid[p.y][p.x] += 1
                    elif path[j+1].x == x and path[j+1].y == y:
                        clabel = "-"
                        grid[p.y][p.x] += 1

                    if grid[p.y, p.x] % 3 == 0 and grid[p.y, p.x] != 0:
                        temp = "\n" + clabel
                        label.append(temp)
                    else:
                        label.append(clabel)

            label = " | ".join(label)
            # visited.append(point)
            row_visit.append(point)
            
            ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                    [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                    facecolor="green", 
                    alpha=0.5)
            ax.text(x+1, y+1, label, ha="center", va="center", color="black", fontsize=8)
            
        
        plt_title = "Adapted A* algorithm:"
        plt.title(plt_title)

        # plt.show()

        file_name = "traj%s.png"%(str(id))
        plt.savefig(os.path.join(dir_path, file_name))
        plt.close()
        