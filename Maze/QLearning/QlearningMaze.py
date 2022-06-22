import numpy as np
import random
import time
import matplotlib
import matplotlib.pyplot as plt

from ql_maze import MazeAI, Direction, Point

from datetime import datetime
import time
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
                #elif self.grid[j][i] == 4:
                #    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'b', alpha=0.75)

        plt_title = "Results of eipsode %s, step %s" %(str(episode), str(step)) 
        plt.title(plt_title)

def plot(rewards, steps, learning_rate, discount_rate, exploration_rate):
    # plt.figure(1)
    # plt.clf() # del if param sweep
    # plt.title('Steps per episode')
    # plt.xlabel('Episode')
    # plt.ylabel('Number of steps')
    
    # plt.plot(steps)
 
    # plt.pause(0.001)
    # plt.draw()

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    avg_reward = 0
    avg_steps = 0

    if len(rewards) != 0: avg_reward = sum(rewards)/len(rewards)
    if len(steps) != 0: avg_steps = sum(steps)/len(steps)

    plt_title = "Results: α=%s, γ=%s, ϵ=%s\nAverage reward: %s\nAverage steps: %s" %(
        str(learning_rate), 
        str(discount_rate), 
        str(exploration_rate), 
        str(avg_reward), 
        str(avg_steps)
        )
    fig.suptitle(plt_title)

    ax[0].plot(np.arange(0, len(rewards)), rewards)
    ax[0].set_title('Rewards per episode')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Rewards')

    ax[1].plot(np.arange(0, len(steps)), steps)
    ax[1].set_title('Steps per episode')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('#Steps')

# Creating The Environment
env = MazeAI()

# Creating The Q-Table
action_space_size = len(Direction)
state_space_size = env.grid.shape[1] * env.grid.shape[0]
#print(state_space_size, action_space_size)

q_table = np.zeros((state_space_size, action_space_size))

# Initializing Q-Learning Parameters
num_episodes = 1000
max_steps_per_episode = 30000

learning_rate = np.array([0.1]) # 0.01
discount_rate = np.array([0.98]) # 0.9

exploration_rate = np.array([1], dtype=np.float32) # 0.01
max_exploration_rate = np.array([1], dtype=np.float32)
min_exploration_rate = np.array([0.01], dtype=np.float32)
exploration_decay_rate = 0.01

# Debug menu
debug_q = input("Default mode: 0\nDebug mode: 1\nSelect mode: ")
debug_flag = int(debug_q)

# Result parameters
episode_cycle = 999
num_sims = len(learning_rate) * len(discount_rate) * len(exploration_rate)
sim = 0

if debug_flag:
    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'Results')
    date_and_time = datetime.now()
    PATH = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(PATH): os.makedirs(PATH)

for lr_i in np.arange(len(learning_rate)):
    for dr_i in np.arange(len(discount_rate)):
        for er_i in np.arange(len(exploration_rate)):
            print("Simulation: ", sim)
            # Reinitialize some variables
            exploration_rate = np.array([1], dtype=np.float32) # 0.01
            q_table = np.zeros((state_space_size, action_space_size))

            # Training Loop
            all_grids = []
            grid_step = []
            episode_len = []
            steps_per_episode = []
            rewards_per_episode = []

            # Q-learning algorithm
            for episode in range(num_episodes):
                if episode % 1 == 0: print("Episode: ", episode)
                
                # Initialize new episode params
                state = env.reset(sim, episode)
                done = False
                rewards_current_episode = 0
                reward = 0

                if debug_flag:
                    if episode % episode_cycle == 0 and episode != 0:
                        all_grids.append(env.grid.copy())
                        grid_step.append(0)
                        #print("Ep: ", episode, ", st: ", step, "\n",env.grid)
                        #print(all_grids)

                for step in range(max_steps_per_episode):
                    #if step % 10000 == 0: print("Step: ", step)
                    # Exploration-exploitation trade-off
                    exploration_rate_threshold = random.uniform(0, 1)
                    if exploration_rate_threshold > exploration_rate[er_i]:
                        action = np.argmax(q_table[state,:])
                        #print(np.argmax(q_table[state,:]), action)
                    else:
                        action_space_actual_size = action_space_size-1
                        action = random.randint(0, action_space_actual_size)
                    
                    #print("Ep: ", episode, "St: ", step)
                    # Take new action
                    new_state, reward, done, info = env.step(action)

                    if debug_flag and episode % episode_cycle == 0 and episode != 0 and not np.array_equiv(state, new_state):
                        all_grids.append(env.grid.copy())
                        grid_step.append(step+1)

                    # if episode % 200 == 0 and episode != 0:
                    #     print("New state:\n", new_state, "\nReward:\n", reward, "\nAction (R,L,U,D):\n", action, "\nMaze:\n", env.grid)
                    #     print("")
                    #if step == max_steps_per_episode-1:
                    #   print("maz limit reached")

                    # Update Q-table
                    q_table[state, action] = q_table[state, action] * (1 - learning_rate[lr_i]) + \
                        learning_rate[lr_i] * (reward + discount_rate[dr_i] * np.max(q_table[new_state, :]))

                    # Set new state
                    state = new_state
                    
                    # Add new reward  
                    rewards_current_episode += reward

                    if done == True:
                        rewards_per_episode.append(rewards_current_episode)
                        steps_per_episode.append(step+1)
                        break

                    # Exploration rate decay 
                    exploration_rate[er_i] = min_exploration_rate[er_i] + \
                        (max_exploration_rate[er_i] - min_exploration_rate[er_i]) * np.exp(-exploration_decay_rate*episode) 
                
                # If the step limit was reached or done is True then the episodes data is saved here
                if debug_flag and not done:
                    if episode % episode_cycle == 0 and episode != 0:
                        all_grids.append(env.grid.copy())
                        grid_step.append(step+1)

            # Add current episode reward to total rewards list
            if debug_flag:
                if not done:
                    rewards_per_episode.append(rewards_current_episode)
                    steps_per_episode.append(step+1)
                plot(rewards_per_episode, steps_per_episode, learning_rate[lr_i], discount_rate[dr_i], exploration_rate[er_i])
                #print(rewards_per_episode)

            print("\n\n********Q-table********\n")
            print(q_table)

            if debug_flag:
                path = os.getcwd()
                path = PATH

                file_name = "learning_curve%s α=%s, γ=%s, ϵ=%s.png" %(str(sim), str(learning_rate[lr_i]), str(discount_rate[dr_i]), str(exploration_rate[er_i]))
                plt.savefig(os.path.join(path, file_name))
                plt.close()

                #print(grid_step)

                #folder = "Episode %s" %(str(episode_cycle))
                #temp_path = os.path.join(path, folder)

                if num_sims == 1:

                    episode = episode_cycle

                    for i in np.arange(len(all_grids)):
                        if grid_step[i] == 0 and i != 0: episode += episode_cycle

                        PR = print_results(all_grids[i], env.grid.shape[0], env.grid.shape[1])
                        
                        # Debug comments
                        # print(episode, " - ", step)
                        # print(all_grids[i])

                        PR.print_graph(episode, grid_step[i])
                        
                        file_name = "plot%s-%s.png" %(str(episode),grid_step[i])
                        
                        #plt.savefig(os.path.join(temp_path, file_name))
                        plt.savefig(os.path.join(path, file_name))

                        plt.close()
            sim += 1

if num_sims == 1:
    nb_success = 0
    # Display percentage successes
    for episode in range(num_episodes):
        if episode % 100 == 0: print(episode)
        # initialize new episode params
        state = env.reset(sim, episode)
        done = False
        for step in range(max_steps_per_episode):        
            # Show current state of environment on screen
            # Choose action with highest Q-value for current state       
            # Take new action
            action = np.argmax(q_table[state,:])  
            new_state, reward, done, info = env.step(action)
            
            if done:
                nb_success += 1
                break       

            # Set new state
            state = new_state

    # Let's check our success rate!
    print (f"Success rate = {nb_success/num_episodes*100}%")