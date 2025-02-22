import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.lines as mlines

import os


from dqn_environment import States, Direction, Point

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
    
    def print_trajectories(self, ax, dir_traj, p, env, actions=None, reward=0, done=None):
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
                text.set_text(f"Drone {i}:\nAction: {action}\nReward: {reward}\nFuel: {env.fuel}\nExplored: {env.explored_from_last}")
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
        plt.pause(0.005)
        if done:
            plt.pause(0.5)

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

    
    def print_graph(self, success, policy, path, actions, starting_pos, obstacles, dir_path, cntr, env):
        """
        Prints the grid environment
        """

        plt.rc('font', size=20)
        plt.rc('axes', titlesize=10)

        # Prints graph
        fig,ax = plt.subplots(figsize=(env.grid.shape[1], env.grid.shape[0]))

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

        for i in range(env.grid.shape[0]): # y
            for j in range(env.grid.shape[1]): # x
                ax.fill([j+0.5, j + 1.5, j + 1.5, j+0.5], [i+0.5, i+0.5, i + 1.5, i + 1.5], facecolor="white", alpha=0.5)
                if obstacles[i,j] == True: ax.fill([j+0.5, j + 1.5, j + 1.5, j+0.5], [i+0.5, i+0.5, i + 1.5, i + 1.5], facecolor="k", alpha=0.5)
                elif Point(j,i) == starting_pos:
                    ax.fill([j + 0.5, j + 1.5, j + 1.5, j + 0.5],\
                            [i + 0.5, i + 0.5, i + 1.5, i + 1.5], \
                                facecolor="red", alpha=0.5)
                    
        # fill explored cells green
        for pos in path:
            x = pos.x
            y = pos.y
            ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
                    [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
                    facecolor="green", 
                    alpha=0.5)
            
        ax.fill([starting_pos.x + 0.5, starting_pos.x + 1.5, starting_pos.x + 1.5, starting_pos.x + 0.5],\
                            [starting_pos.y + 0.5, starting_pos.y + 0.5, starting_pos.y + 1.5, starting_pos.y + 1.5], \
                                facecolor="red", alpha=0.5)
        
        ax.fill([path[-1].x + 0.5, path[-1].x + 1.5, path[-1].x + 1.5, path[-1].x + 0.5],\
                            [path[-1].y + 0.5, path[-1].y + 0.5, path[-1].y + 1.5, path[-1].y + 1.5], \
                                facecolor="blue", alpha=0.5)
            
        # adds all indices of actions on cell
        indices = {}
        for x in range(env.grid.shape[0]): # y
            for y in range(env.grid.shape[1]): # x
                for i, pos in enumerate(path):
                    if pos == Point(x,y): # if cell in path then add index to dict
                        if pos in indices: # checks if dict already has key named "pos"
                            indices[pos].append(i)
                        else:
                            indices[pos] = [i]
        
        cell_count = 0
        clabel = ""
        for x in range(env.grid.shape[0]): # y
            for y in range(env.grid.shape[1]): # x
                if Point(x,y) in path:
                    for i in indices[Point(x,y)]:
                        if actions[i] == Direction.RIGHT.value: 
                            clabel += "\u2192"
                            breakpoint
                        elif actions[i] == Direction.LEFT.value: 
                            clabel += "\u2190"
                            breakpoint
                        elif actions[i] == Direction.UP.value: 
                            clabel += "\u2193"
                            breakpoint
                        elif actions[i] == Direction.DOWN.value: 
                            clabel += "\u2191"
                            breakpoint

                temp_label = ""
                if len(clabel) > 3:
                    for j, c in enumerate(clabel):
                        temp_label += clabel[j:j+8] + "\n"
                    clabel = temp_label
                ax.text(x+1, y+1, clabel, ha="center", va="center", color="black", fontsize=8)
                clabel = ""
                # plt.pause(0.005)
                

        # # Add path to plot
        # temp_grid_ = np.zeros(env.grid.shape)
        # for i in visited:
        #     x, y = i.x, i.y
        #     if temp_grid_[y][x] < 1:
        #         ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5],
        #             [y + 0.5, y + 0.5, y + 1.5, y + 1.5],
        #             facecolor="y",
        #             alpha=0.5)
        #         temp_grid_[y][x] += 1

        # clabel = ""
        # grid = np.zeros((env.grid.shape[0], env.grid.shape[1]))
        # row_visit = []
        # for i, point in enumerate(path):
        #     if i == len(path)-1: break
        #     x, y = point.x, point.y
        #     label = []
        #     for j, p in enumerate(path):
        #         if p == point and p not in row_visit:
        #             if j == len(path)-1: break
        #             if path[j+1].x == x+1:
        #                 clabel = ">"
        #                 grid[p.y][p.x] += 1
        #             elif path[j+1].x == x-1:
        #                 clabel = "<"
        #                 grid[p.y][p.x] += 1
        #             elif path[j+1].y == y+1:
        #                 clabel = "^"
        #                 grid[p.y][p.x] += 1
        #             elif path[j+1].y == y-1:
        #                 clabel = "v"
        #                 grid[p.y][p.x] += 1
        #             elif path[j+1].x == x and path[j+1].y == y:
        #                 clabel = "-"
        #                 grid[p.y][p.x] += 1

        #             if grid[p.y, p.x] % 3 == 0 and grid[p.y, p.x] != 0:
        #                 temp = clabel + "\n"
        #                 label.append(temp)
        #             else:
        #                 label.append(clabel)

        #     label = " | ".join(label)
        #     visited.append(point)
        #     row_visit.append(point)

        #     if temp_grid_[y][x] < 2 and not Point(x,y) == starting_pos:
        #         ax.fill([x + 0.5, x + 1.5, x + 1.5, x + 0.5], 
        #                 [y + 0.5, y + 0.5, y + 1.5, y + 1.5], 
        #                 facecolor="green", 
        #                 alpha=0.5)
        #         temp_grid_[y][x] += 1
        #     if not i == len(path)-2:
        #         ax.text(x+1, y+1, label, ha="center", va="center", color="black", fontsize=8)
            
        plt_title = "DQN algorithm:\nSuccess: " + str(success)
        plt.title(plt_title)

        file_name = "traj%s_%s.png"%(str(policy), str(cntr))
        plt.savefig(os.path.join(dir_path, file_name))
        plt.close()     