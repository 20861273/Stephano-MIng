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
    def print_graph(self, episode, step):
        """
        Prints the grid environment
        """

        plt.rc('font', size=12)
        plt.rc('axes', titlesize=15) 

        # Prints graph
        fig,ax = plt.subplots(figsize=(8, 8))

        # Set tick locations
        ax.set_xticks(np.arange(-0.5, self.cols*2+0.5, step=2),minor=False)
        ax.set_yticks(np.arange(-0.5, self.rows*2+0.5, step=2),minor=False)
        
        plt.xticks(rotation=90)
    
        xticks = list(map(str,np.arange(0, self.cols+1, step=1)))
        yticks = list(map(str,np.arange(0, self.rows+1, step=1)))
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)

        # Set grid
        plt.grid(which='major',axis='both', color='k')

        # Print
        for j in range(self.rows):
            for i in range(self.cols):
                x1 = (i-0.5)*2 + 0.5
                x2 = (i+0.5)*2 + 0.5
                y1 = (self.rows - (j-0.5) - 1)*2 + 0.5
                y2 = (self.rows - (j+0.5) - 1)*2 + 0.5
                if self.grid[j][i] == 0:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'w', alpha=0.75)
                elif self.grid[j][i] == 1:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'b', alpha=0.75)
                elif self.grid[j][i] == 2:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'g', alpha=0.75)
                elif self.grid[j][i] == 3:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'r', alpha=0.75)
                elif self.grid[j][i] == 4:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'y', alpha=0.75)

        plt_title = "Q-learning Results: Episode %s, step %s" %(str(episode), str(step)) 
        plt.title(plt_title)

    def plot(self, q_tables, rewards, steps, learning_rate, discount_rate, exploration_rate, save_path, env):
        f = open(os.path.join(save_path,"saved_data.txt"), "w", encoding="utf-8")

        fig, ax = plt.subplots(1, 2, figsize=(30, 15))
        
        ax[0].set_title('Rewards per episode')
        ax[0].set_xlabel('Episode')
        ax[0].set_ylabel('Rewards')

        ax[1].set_title('Steps per episode')
        ax[1].set_xlabel('Episode')
        ax[1].set_ylabel('#Steps')

        c = cm.rainbow(np.linspace(0, 1, len(rewards)))
        l = []
        cnt = 0

        f.write(str("Starting position: %s\nGoal: %s" %(str(env.starting_pos), str(env.goal))))
        for lr_i in np.arange(len(learning_rate)):
            for dr_i in np.arange(len(discount_rate)):
                for er_i in np.arange(len(exploration_rate)):
                    file_name = "policy" + str(cnt) + ".txt"
                    np.savetxt(os.path.join(save_path, file_name), q_tables[cnt])
                    
                    l.append("%s: α=%s, γ=%s, ϵ=%s" %(
                            str(cnt),
                            str(learning_rate[lr_i]), 
                            str(discount_rate[dr_i]), 
                            str(exploration_rate[er_i])
                            ))
                    cnt += 1      
        
        f.close()

        file_name = "maze.txt"
        np.savetxt(os.path.join(save_path, file_name), env.grid)

        for i in range(0, len(rewards)):
            ax[0].plot(np.arange(0, len(rewards[i])), rewards[i], color=c[i])
            ax[1].plot(np.arange(0, len(steps[i])), steps[i], color=c[i])

        ax[0].legend(l)
        ax[1].legend(l)

        file_name = "learning_curve.png"
        plt.savefig(os.path.join(save_path, file_name))
        plt.close()
        