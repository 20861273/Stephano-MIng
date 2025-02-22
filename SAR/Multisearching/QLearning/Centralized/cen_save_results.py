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
    def print_graph(self, step):
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

        plt_title = "Q-learning Results: Step %s" %(str(step)) 
        plt.title(plt_title)

    def plot_and_save(self, q_tables, rewards, pos_reward, epochs,
                    learning_rate, discount_rate, exploration_rate,
                    min_exploration_rate, max_exploration_rate, exploration_decay_rate,
                    save_path, env, h, m, s, interval, trajs0, trajs1, qtrajs0):
        f = open(os.path.join(save_path,"env_shape.txt"), "w", encoding="utf-8")

        c = cm.rainbow(np.linspace(0, 1, len(rewards)))
        leg = []
        cnt = 0

        f.write(str(env.grid.shape))
        for lr_i in np.arange(len(learning_rate)):
            for dr_i in np.arange(len(discount_rate)):
                for er_i in np.arange(len(exploration_rate)):
                    for pos_i in np.arange(len(pos_reward)):
                        file_name = "policy" + str(cnt) + ".txt"
                        np.savetxt(os.path.join(save_path, file_name), q_tables[cnt])
                        
                        leg.append("%s: r=%s, α=%s, γ=%s, ϵ=%s, ϵ_min=%s, ϵ_max=%s, ϵ_d=%s" %(
                                str(cnt),
                                str(pos_reward[pos_i]), 
                                str(learning_rate[lr_i]), 
                                str(discount_rate[dr_i]), 
                                str(exploration_rate[er_i]),
                                str(min_exploration_rate[er_i]),
                                str(max_exploration_rate[er_i]),
                                str(exploration_decay_rate[er_i])
                                ))
                        cnt += 1      
        
        f.close()

        for i in range(0, len(rewards)):
            file_name = "policy_rewards" + str(i) + ".txt"
            np.savetxt(os.path.join(save_path, file_name), rewards[i])

        f = open(os.path.join(save_path,"trajectories0.txt"), "w", encoding="utf-8")

        line = ""
        for i in range(len(trajs0)):
            for j in range(0, env.grid.shape[0]):
                f.write('\n')
                for k in range(0, env.grid.shape[0]):
                    for l in range(0, env.grid.shape[1]):
                        line += str(trajs0[i][j][l][k]) + "  "
                    f.write(str(line))
                    line = ""
                    f.write('\n')
            f.write('\n\n')

        f.close()

        f = open(os.path.join(save_path,"trajectories1.txt"), "w", encoding="utf-8")

        line = ""
        for i in range(len(trajs1)):
            for j in range(0, env.grid.shape[0]):
                f.write('\n')
                for k in range(0, env.grid.shape[0]):
                    for l in range(0, env.grid.shape[1]):
                        line += str(trajs1[i][j][l][k]) + "  "
                    f.write(str(line))
                    line = ""
                    f.write('\n')
            f.write('\n\n')

        f.close()

        f = open(os.path.join(save_path,"qtrajectories0.txt"), "w", encoding="utf-8")

        line = ""
        for i in range(len(qtrajs0)):
            for j in range(0, env.grid.shape[0]):
                f.write('\n')
                for k in range(0, env.grid.shape[0]):
                    for l in range(0, env.grid.shape[1]):
                        line += str(qtrajs0[i][j][l][k]) + "  "
                    f.write(str(line))
                    line = ""
                    f.write('\n')
            f.write('\n\n')

        f.close()
        
        sim_len = (len(learning_rate) * len(discount_rate) * len(exploration_rate) * len(pos_reward))
        plot_len = int(sim_len/3)
        plot_rem = sim_len % 3
        cnt = 0
        for i in range(0, plot_len):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

            ax1.set_title('Rewards per episode\nTraining time: %sh %sm %ss\nEpochs: %s' %(h, m, s, epochs))
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Rewards')

            ax2.set_title('Rewards per episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('#Steps')

            ax1.set_ylim([-10, 10])
            ax2.set_ylim([-10, 10])

            for j in range(0, 3):
                ax1.plot(np.arange(0, len(rewards[i*3+j][0]))*interval, rewards[i*3+j][0], color=c[i*3+j])
                ax2.plot(np.arange(0, len(rewards[i*3+j][1]))*interval, rewards[i*3+j][1], color=c[i*3+j])
                cnt += 1

            ax1.legend(leg[i*3:i*3+3])
            ax2.legend(leg[i*3:i*3+3])

            file_name = "learning_curve" + str(i) + ".png"
            plt.savefig(os.path.join(save_path, file_name))
            plt.close()

        if plot_rem != 0:
            fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(30, 15))

            ax1.set_title('Rewards per episode\nTraining time: %sh %sm %ss\nEpochs: %s' %(h, m, s, epochs))
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Rewards')

            ax2.set_title('Rewards per episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('#Steps')

            ax1.set_ylim([-10, 10])
            ax2.set_ylim([-10, 10])

            for i in range(sim_len-plot_rem, sim_len):
                ax1.plot(np.arange(0, len(rewards[i][0]))*interval, rewards[i][0], color=c[i])
                ax2.plot(np.arange(0, len(rewards[i][1]))*interval, rewards[i][1], color=c[i])

            ax1.legend(leg[cnt:])
            ax2.legend(leg[cnt:])

            file_name = "learning_curve" + str(i) + ".png"
            plt.savefig(os.path.join(save_path, file_name))
            plt.close()
        