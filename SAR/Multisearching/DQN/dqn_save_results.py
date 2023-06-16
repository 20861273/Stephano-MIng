import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.lines as mlines

import os

from dqn_environment import States, Direction

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
    # def print_graph(self, step):
    #     """
    #     Prints the grid environment
    #     """

    #     plt.rc('font', size=12)
    #     plt.rc('axes', titlesize=15) 

    #     # Prints graph
    #     fig,ax = plt.subplots(figsize=(8, 8))

    #     # Set tick locations
    #     ax.set_xticks(np.arange(-0.5, self.cols*2+0.5, step=2),minor=False)
    #     ax.set_yticks(np.arange(-0.5, self.rows*2+0.5, step=2),minor=False)
        
    #     plt.xticks(rotation=90)
    
    #     xticks = list(map(str,np.arange(0, self.cols+1, step=1)))
    #     yticks = list(map(str,np.arange(0, self.rows+1, step=1)))
    #     ax.set_xticklabels(xticks)
    #     ax.set_yticklabels(yticks)

    #     # Set grid
    #     plt.grid(which='major',axis='both', color='k')

    #     # Print
    #     for j in range(self.rows):
    #         for i in range(self.cols):
    #             x1 = (i-0.5)*2 + 0.5
    #             x2 = (i+0.5)*2 + 0.5
    #             y1 = (self.rows - (j-0.5) - 1)*2 + 0.5
    #             y2 = (self.rows - (j+0.5) - 1)*2 + 0.5
    #             if self.grid[j][i] == 0:
    #                 plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'w', alpha=0.75)
    #             elif self.grid[j][i] == 1:
    #                 plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'b', alpha=0.75)
    #             elif self.grid[j][i] == 2:
    #                 plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'g', alpha=0.75)
    #             elif self.grid[j][i] == 3:
    #                 plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'r', alpha=0.75)
    #             elif self.grid[j][i] == 4:
    #                 plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'y', alpha=0.75)

    #     plt_title = "DQN Results: Step %s" %(str(step)) 
    #     plt.title(plt_title)
    def get_actions(self, trajectory, y, x):
        actions = []
        for p, act, r in trajectory:
            if (p.x,p.y) == (x,y):
                actions.append((r, act))
        return actions
    
    def print_trajectories(self, ax, dir_traj, p, env, actions=None, reward=0, done=False):
        """
        Prints the grid environment
        """
        # plt.clf()
        scale = 2
        plt.ion()
        plt.rc('font', size=20)
        plt.rc('axes', titlesize=10)

        # Prints graph        

        ax.set_aspect("equal")
        ax.set_xlim(0.5*scale, self.cols + 0.5*scale)
        ax.set_ylim(0.5*scale, self.rows + 0.5*scale)
        # Set tick positions to be centered between grid lines
        ax.set_xticks(np.arange(self.cols) + 0.5*scale)
        ax.set_yticks(np.arange(self.rows) + 0.5*scale)

        # Set tick labels to be the x or y coordinate of the grid cell
        ax.set_xticklabels(np.arange(self.cols))
        ax.set_yticklabels(np.arange(self.rows))

        # Adjust tick label position and font size
        ax.tick_params(axis='both', labelsize=10, pad=2, width=0.5*scale, length=2)
        ax.grid(True, color='black', linewidth=1)

        temp_grid = env.grid.copy()
        grid = temp_grid[::-1]

        drone_locations = []
        for i_r in range(env.nr):
            drone_locations.append((grid.shape[0] - env.pos[i_r].y - 1, env.pos[i_r].x))            

        for j in range(self.cols):
            for i in range(self.rows):
                if grid[i][j] == States.OBS.value:
                    ax.fill([j + 0.5*scale, j + 1.5*scale, j + 1.5*scale, j + 0.5*scale],\
                            [i + 0.5*scale, i + 0.5*scale, i + 1.5*scale, i + 1.5*scale], \
                                facecolor="k", alpha=0.5*scale)
                elif grid[i][j] == States.EXP.value:
                    if done:
                        ax.fill([j + 0.5*scale, j + 1.5*scale, j + 1.5*scale, j + 0.5*scale],\
                                [i + 0.5*scale, i + 0.5*scale, i + 1.5*scale, i + 1.5*scale], \
                                    facecolor="yellow", alpha=0.5*scale)
                    else:
                        ax.fill([j + 0.5*scale, j + 1.5*scale, j + 1.5*scale, j + 0.5*scale],\
                                [i + 0.5*scale, i + 0.5*scale, i + 1.5*scale, i + 1.5*scale], \
                                    facecolor="green", alpha=0.5*scale)
                elif grid[i][j] == States.GOAL.value:
                    ax.fill([j + 0.5*scale, j + 1.5*scale, j + 1.5*scale, j + 0.5*scale],\
                            [i + 0.5*scale, i + 0.5*scale, i + 1.5*scale, i + 1.5*scale], \
                                facecolor="red", alpha=0.5*scale)
                    # ax.fill([j + 0.5*scale, j + 1.5*scale, j + 1.5*scale, j + 0.5*scale],\
                    #         [i + 0.5*scale, i + 0.5*scale, i + 1.5*scale, i + 1.5*scale], \
                    #             facecolor="white", alpha=0.5*scale)
                elif grid[i][j] == States.ROBOT.value:
                    blues = np.linspace(0.3, 1, env.nr)
                    for r_i in range(env.nr):
                        color = (0,0,blues[r_i])
                        if (drone_locations[r_i]) == (i,j):
                            ax.fill([j + 0.5*scale, j + 1.5*scale, j + 1.5*scale, j + 0.5*scale],\
                                    [i + 0.5*scale, i + 0.5*scale, i + 1.5*scale, i + 1.5*scale], \
                                        facecolor=color, alpha=0.5*scale)
                elif grid[i][j] == States.UNEXP.value:
                    ax.fill([j + 0.5*scale, j + 1.5*scale, j + 1.5*scale, j + 0.5*scale], \
                            [i + 0.5*scale, i + 0.5*scale, i + 1.5*scale, i + 1.5*scale], \
                                facecolor="white", alpha=0.5*scale)
        
        
        s = str(p)
        plt_title = "DQN policy: " + s
        plt.title(plt_title)

        if actions != None:
            legend = ax.legend(actions, loc='center left', bbox_to_anchor=(1, 0.5))
            handles = legend.legendHandles

            blues = np.linspace(0.3, 1, env.nr)
            for i,handle in enumerate(handles):
                color = (0,0,blues[i])
                handle.set_facecolor(color)

            for i,text in enumerate(legend.get_texts()):
                if actions[i] == Direction.LEFT.value:
                    action = "Left [<]"
                elif actions[i] == Direction.RIGHT.value:
                    action = "Right [>]"
                elif actions[i] == Direction.UP.value:
                    action = "Up [^]"
                elif actions[i] == Direction.DOWN.value:
                    action = "Down [v]"
                text.set_text(f"Drone {i}:\nAction: {action}\nReward: {reward}")
        else:
            legend = ax.legend([0]*env.nr, loc='center left', bbox_to_anchor=(1, 0.5))
            handles = legend.legendHandles

            blues = np.linspace(0.3, 1, env.nr)
            for i,handle in enumerate(handles):
                color = (0,0,blues[i])
                handle.set_facecolor(color)

            for i,text in enumerate(legend.get_texts()):
                text.set_text(f"Drone {i}:")

            

        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()

        # file_name = "p%dtrajectory%d.png" %(policy, cnt)
        # plt.savefig(os.path.join(dir_traj, file_name))
        plt.pause(0.05)
        if done: plt.pause(0.5)

    def plot(self, rewards, steps, learning_rate, discount_rate, exploration_rate, save_path, env, t_time, postive_reward):
        f = open(os.path.join(save_path,"saved_data.txt"), "w", encoding="utf-8")

        m, s = divmod(t_time, 60)
        h = 0
        if m >= 60: h, m = divmod(m, 60)

        c = cm.rainbow(np.linspace(0, 1, len(rewards)))
        l = []
        cnt = 0

        f.write(str(env.grid.shape))
        for lr_i in np.arange(len(learning_rate)):
            for dr_i in np.arange(len(discount_rate)):
                for er_i in np.arange(len(exploration_rate)):                    
                    l.append("%s: α=%s, γ=%s, ϵ=%s" %(
                            str(cnt),
                            str(learning_rate[lr_i]), 
                            str(discount_rate[dr_i]), 
                            str(exploration_rate[er_i])
                            ))
                    cnt += 1      
        
        f.close()

        for i in range(0, len(rewards)):
            file_name = "policy_rewards" + str(i) + ".txt"
            np.savetxt(os.path.join(save_path, file_name), rewards[i])
        
        sim_len = (len(learning_rate) * len(discount_rate) * len(exploration_rate))
        plot_len = int(sim_len/3)
        plot_rem = sim_len % 3
        cnt = 0
        for i in range(0, plot_len):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

            ax1.set_title('Rewards per episode\nTraining time: %sh %sm %ss\nPositive reward: %s' %(h, m, s, str(postive_reward)))
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Rewards')

            ax2.set_title('Steps per episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('#Steps')
            
            for j in range(0, 3):
                ax1.plot(np.arange(0, len(rewards[i*3+j]), 5), rewards[i*3+j][::5], color=c[i*3+j])
                ax2.plot(np.arange(0, len(steps[i*3+j]), 5), steps[i*3+j][::5], color=c[i*3+j])
                cnt += 1

            ax1.legend(l[i*3:i*3+3])
            ax2.legend(l[i*3:i*3+3])

            file_name = "learning_curve" + str(i) + ".png"
            plt.savefig(os.path.join(save_path, file_name))
            plt.close()

        if plot_rem != 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

            ax1.set_title('Rewards per episode\nTraining time: %sh %sm %ss' %(h, m, s))
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Rewards')

            ax2.set_title('Steps per episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('#Steps')

            for i in range(sim_len-plot_rem, sim_len):
                ax1.plot(np.arange(0, len(rewards[i]), 5), rewards[i][::5], color=c[i])
                ax2.plot(np.arange(0, len(steps[i]), 5), steps[i][::5], color=c[i])

            ax1.legend(l[cnt:])
            ax2.legend(l[cnt:])

            file_name = "learning_curve" + str(i) + ".png"
            plt.savefig(os.path.join(save_path, file_name))
            plt.close()

    
        