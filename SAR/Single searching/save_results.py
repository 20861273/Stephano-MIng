import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from environment import Direction, Point, States

import os
import json
import math

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
        drone_locations.append((grid.shape[0] - env.pos.y - 1, env.pos.x))            

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
                    blues = np.linspace(0.3, 1, 1)
                    for r_i in range(1):
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

        # if actions != None:
        #     legend = ax.legend(actions, loc='center left', bbox_to_anchor=(1, 0.5))
        #     handles = legend.legendHandles

        #     blues = np.linspace(0.3, 1, env.nr)
        #     for i,handle in enumerate(handles):
        #         color = (0,0,blues[i])
        #         handle.set_facecolor(color)

        #     for i,text in enumerate(legend.get_texts()):
        #         if actions[i] == Direction.LEFT.value:
        #             action = "Left [<]"
        #         elif actions[i] == Direction.RIGHT.value:
        #             action = "Right [>]"
        #         elif actions[i] == Direction.UP.value:
        #             action = "Up [^]"
        #         elif actions[i] == Direction.DOWN.value:
        #             action = "Down [v]"
        #         text.set_text(f"Drone {i}:\nAction: {action}\nReward: {reward}")
        #         # if actions[i][0] == Direction.LEFT.value:
        #         #     action = "Left [<]"
        #         # elif actions[i][0] == Direction.RIGHT.value:
        #         #     action = "Right [>]"
        #         # # elif actions[i] == Direction.UP.value:
        #         # #     action = "Up [^]"
        #         # # elif actions[i] == Direction.DOWN.value:
        #         # #     action = "Down [v]"
        #         # elif actions[i][0] == Direction.FORWARD.value:
        #         #     action = "FORWARD"
        #         # text.set_text(f"Drone {i}:\nAction: {action}\nReward: {reward}")
        # else:
        #     legend = ax.legend([0]*env.nr, loc='center left', bbox_to_anchor=(1, 0.5))
        #     handles = legend.legendHandles

        #     blues = np.linspace(0.3, 1, env.nr)
        #     for i,handle in enumerate(handles):
        #         color = (0,0,blues[i])
        #         handle.set_facecolor(color)

        #     for i,text in enumerate(legend.get_texts()):
        #         text.set_text(f"Drone {i}:")

            

        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()

        # file_name = "p%dtrajectory%d.png" %(policy, cnt)
        # plt.savefig(os.path.join(dir_traj, file_name))
        plt.pause(0.0005)
        if done:
            plt.pause(0.5)

    def plot(self, q_tables, rewards, steps, learning_rate, discount_rate, exploration_rate, save_path, env, t_time, trajs, epochs):
        f = open(os.path.join(save_path,"saved_data.txt"), "w", encoding="utf-8")

        c = cm.rainbow(np.linspace(0, 1, len(rewards)))
        l = []
        cnt = 0

        f.write(str(env.grid.shape))
        for lr_i in np.arange(len(learning_rate)):
            for dr_i in np.arange(len(discount_rate)):
                for er_i in np.arange(len(exploration_rate)):
                    for e in np.arange(epochs):
                        file_name = "policy" + str(cnt) + "_" + str(e)
                        np.save(os.path.join(save_path, file_name), q_tables[cnt, e])
                        
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

        # for i in range(0, len(trajs)):
        #     file_name = "trajectories" + str(i) + ".txt"
        #     np.savetxt(os.path.join(save_path, file_name), trajs[i], fmt='%s')
        
        sim_len = (len(learning_rate) * len(discount_rate) * len(exploration_rate))
        plot_len = int(sim_len/3)
        plot_rem = sim_len % 3
        for i in range(0, plot_len):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

            ax1.set_title('Rewards per episode\nTraining time: %sm %ss' %(divmod(t_time, 60)))
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Rewards')

            ax2.set_title('Steps per episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('#Steps')
            
            for j in range(0, 3):
                ax1.plot(np.arange(0, len(rewards[i*3+j]), 1), rewards[i*3+j][::1], color=c[i*3+j])
                ax2.plot(np.arange(0, len(steps[i*3+j]), 1), steps[i*3+j][::1], color=c[i*3+j])

            ax1.legend(l)
            ax2.legend(l)

            file_name = "learning_curve" + str(i) + ".png"
            plt.savefig(os.path.join(save_path, file_name))
            plt.close()

        if plot_rem != 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15))

            ax1.set_title('Rewards per episode\nTraining time: %sm %ss' %(divmod(t_time, 60)))
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Rewards')

            ax2.set_title('Steps per episode')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('#Steps')

            for i in range(sim_len-plot_rem, sim_len):
                ax1.plot(np.arange(0, len(rewards[i]), 1), rewards[i][::1], color=c[i])
                ax2.plot(np.arange(0, len(steps[i]), 1), steps[i][::1], color=c[i])

            ax1.legend(l)
            ax2.legend(l)

            file_name = "learning_curve" + str(i) + ".png"
            plt.savefig(os.path.join(save_path, file_name))
            plt.close()
    
def plot_learning_curve(nr, training_type, scores, filename, lr, dr, er, pr, negr, per, nsr, ms, totle_time):
    mean_rewards = np.zeros((len(scores[0]),))
    std_rewards = np.zeros((len(scores[0]),))

    if len(scores) > 1:
        for i_ep in range(len(scores[0])):
            s = sum(scores[e][i_ep] for e in range(len(scores)))
            mean_rewards[i_ep] = s / len(scores)
    else:
        mean_rewards = scores[0]

    if len(scores) > 1:
        for i_ep in range(len(scores[0])):
            v = sum((scores[e][i_ep]-mean_rewards[i_ep])**2 for e in range(len(scores)))
            std_rewards[i_ep] = math.sqrt(v / (len(scores)-1))

    fig=plt.figure()
    if training_type == "Q-learning":
        l = "α=%s,\nγ=%s,\nϵ=%s,\npositive reward=%s,\nnegative reward=%s,\npositive eexploration reward=%s,\nnegative step reward=%s,\nmax steps=%s" %(str(lr), str(dr), str(er), str(pr), str(negr), str(per), str(nsr), str(ms))
    elif training_type == "decentralized":
        l = "drone=%s\nα=%s,\nγ=%s,\nϵ=%s,\npositive reward=%s,\nnegative reward=%s,\npositive eexploration reward=%s,\nnegative step reward=%s,\nmax steps=%s" %(str(nr), str(lr), str(dr), str(er), str(pr), str(negr), str(per), str(nsr), str(ms))
    ax=fig.add_subplot(111)

    ax.plot(np.arange(0, len(mean_rewards), 1), mean_rewards[::1], color="C1", label=l)
    if len(scores) > 1:
        plt.fill_between(np.arange(0, len(mean_rewards), int(1)), \
            mean_rewards[::int(1)]-std_rewards[::int(1)], mean_rewards[::int(1)]+std_rewards[::int(1)], alpha = 0.1, color = 'b')
    lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.legend()
    ax.set_xlabel("Episodes", color="C0")
    ax.set_ylabel("Rewards", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    ax.set_ylim(np.array(scores).min()-1, np.array(scores).max()+1)
    m, s = divmod(totle_time, 60)
    h = 0
    if m >= 60: h, m = divmod(m, 60)
    ax.set_title("Learning curve:\nTime: %sh%sm%ss" %(str(h), str(m), str(s)), fontsize = 10)

    plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')

def write_json(lst, file_name):
    with open(file_name, "w") as f:
        json.dump(lst, f)
        