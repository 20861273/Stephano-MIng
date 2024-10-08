import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from ql_maze import WIDTH, MazeAI, Direction, Point

from datetime import datetime
import time
import os

# Creating The Environment
env = MazeAI()

# Creating The Q-Table
action_space_size = len(Direction)
state_space_size = env.grid.shape[1] * env.grid.shape[0]
#print(state_space_size, action_space_size)

q_table = np.zeros((state_space_size, action_space_size))

# Initializing Q-Learning Parameters
num_episodes = 1000
max_steps_per_episode = 10000
num_sequences = 1

learning_rate = np.array([0.1, 0.001]) # 0.01
discount_rate = np.array([0.99]) # 0.9

exploration_rate = np.array([0.01], dtype=np.float32) # 0.01
max_exploration_rate = np.array([0.01], dtype=np.float32)
min_exploration_rate = np.array([0.01], dtype=np.float32)
exploration_decay_rate = np.array([0.01], dtype=np.float32)

# Debug menu
debug_q = input("Default training mode: 1\nGreedy policy: 2\nSelect mode: ")
# debug_q = input("Default training mode: 0\nDebug training mode: 1\nSelect mode: ")
debug_flag = int(debug_q)

# Result parameters
num_sims = len(learning_rate) * len(discount_rate) * len(exploration_rate)
sim = 0
seq = 0
steps_per_episode = []
rewards_per_episode = []
seq_rewards = []
seq_steps = []
avg_rewards = []
avg_steps = []
all_rewards = []
all_steps = []
all_grids = []
q_tables = np.zeros((num_sims, state_space_size, action_space_size))

print("\n\nMaze: ", env.grid.shape, "\n# Training sessions: ", num_sequences, "\n# Simulations per training session: ", num_sims, "\n# Episodes per simulation: ", num_episodes)
print("Hyperparameters:\nLearning rate (α): ", learning_rate, "\nDiscount rate (γ): ", discount_rate, "\nExploration rate (ϵ): ", exploration_rate, "\nExploration decay rate: ", exploration_decay_rate)

if debug_flag != 0:
    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'Results')
    date_and_time = datetime.now()
    save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(save_path): os.makedirs(save_path)
if debug_flag == 2:
    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'Results')
    load_path = os.path.join(PATH, 'saved_data')
    if not os.path.exists(load_path): os.makedirs(load_path)

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

        plt_title = "Q-learning Results: Episode %s, step %s" %(str(episode), str(step)) 
        plt.title(plt_title)

def plot(q_tables, rewards, steps, learning_rate, discount_rate, exploration_rate):
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

    f.write(str("\nStarting position: %s\nExit: %s\nMaze:\n%s" %(str(env.starting_pos), str(env.exit), str(env.grid))))
    for lr_i in np.arange(len(learning_rate)):
        for dr_i in np.arange(len(discount_rate)):
            for er_i in np.arange(len(exploration_rate)):
                #print(q_tables[cnt])
                # f.write(str("\nPolicy %s:\nα=%s, γ=%s, ϵ=%s\n" %(
                #                                                 str(cnt),
                #                                                 str(learning_rate[lr_i]),
                #                                                 str(discount_rate[dr_i]),
                #                                                 str(exploration_rate[er_i]))
                #                                                 ))
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

    for i in range(0, len(rewards)):
        ax[0].plot(np.arange(0, len(rewards[i])), rewards[i], color=c[i])
        ax[1].plot(np.arange(0, len(steps[i])), steps[i], color=c[i])

    ax[0].legend(l)
    ax[1].legend(l)

    file_name = "learning_curve.png"
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()

# Calculates average rewards and steps
def calc_avg(rewards, steps, num_sequences, sim_num, ep_num):
    avg_rewards = np.zeros((sim_num, ep_num))
    avg_steps = np.zeros((sim_num, ep_num))

    for sim_i in range(0, sim_num):
        for ep_i in range(0, ep_num):
            for seq_i in range(0, num_sequences):
                avg_rewards[sim_i, ep_i] += np.array(rewards)[seq_i, sim_i, ep_i]
                avg_steps[sim_i, ep_i] += np.array(steps)[seq_i, sim_i, ep_i]

    for i in range(len(avg_rewards)):
        avg_rewards[i] = avg_rewards[i]/num_sequences
        avg_steps[i] = avg_steps[i]/num_sequences

    return avg_rewards.tolist(), avg_steps.tolist()

def extract_values(correct_path, policy):
    maze = []
    f = open(os.path.join(correct_path,"saved_data.txt"), "r")
    lines = f.readlines()
    maze_flag = False

    for line in lines:
        cur_num = ''
        cur_line = []
        if line[0:18] == "Starting position:":
            for num in line:
                if num.isdigit():
                    cur_num += num
                else:
                    if cur_num != '': cur_line.append(int(cur_num))
                    cur_num = ''
            env.starting_pos = Point(int(cur_line[0]),int(cur_line[1]))

        cur_num = ''
        cur_line = []
        if line[0:5] == "Exit:":
            for num in line:
                if num.isdigit():
                    cur_num += num
                else:
                    if cur_num == ',':
                        cur_line = []
                    elif cur_num != '': cur_line.append(int(cur_num))
                    cur_num = ''
            env.exit = Point(int(cur_line[0]),int(cur_line[1]))
        
        cur_num = ''
        cur_line = []
        if line[0:5] == "Maze:" or maze_flag:
            maze_flag = True
            for num in line:
                if num.isdigit():
                    cur_num += num
                else:
                    if cur_num != '':
                        cur_line.append(int(cur_num))
                    cur_num = ''
            if len(cur_line) == WIDTH: maze.append(cur_line) 

    file_name = "policy" + str(policy) + ".txt"
    return np.loadtxt(os.path.join(correct_path, file_name)), np.array(maze)

if debug_flag != 2:
    # Training loop
    for seq_i in range(0, num_sequences):
        print("Training session: ", seq)
        seq_rewards = []
        seq_steps = []
        sim = 0
        for lr_i in np.arange(len(learning_rate)):
            for dr_i in np.arange(len(discount_rate)):
                for er_i in np.arange(len(exploration_rate)):
                    print("Simulation: ", sim)
                    # Reinitialize some variables
                    ep_exploration_rate = np.copy(exploration_rate)
                    q_table = np.zeros((state_space_size, action_space_size))

                    # Training Loop
                    episode_len = []
                    steps_per_episode = []
                    rewards_per_episode = []                

                    # Q-learning algorithm
                    for episode in range(num_episodes):
                        # print("Episode: ", episode)
                        
                        # Initialize new episode params
                        state = env.reset(sim, episode)
                        done = False
                        rewards_current_episode = 0
                        reward = 0

                        for step in range(max_steps_per_episode):
                            # Exploration-exploitation trade-off
                            exploration_rate_threshold = random.uniform(0, 1)
                            if exploration_rate_threshold > ep_exploration_rate[er_i]:
                                action = np.argmax(q_table[state,:])
                            else:
                                action_space_actual_size = action_space_size-1
                                action = random.randint(0, action_space_actual_size)

                            # Take new action
                            new_state, reward, done, info = env.step(action)

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
                            ep_exploration_rate[er_i] = min_exploration_rate[er_i] + \
                                (max_exploration_rate[er_i] - min_exploration_rate[er_i]) * np.exp(-exploration_decay_rate[er_i]*episode)

                        # Add current episode reward to total rewards list
                        if not done:
                            rewards_per_episode.append(rewards_current_episode)
                            steps_per_episode.append(step+1)

                    
                    tmp_seq_rewards = np.array(seq_rewards)
                    new_tmp_seq_rewards = np.array(np.append(tmp_seq_rewards.ravel(),np.array(rewards_per_episode)))
                    if tmp_seq_rewards.shape[0] == 0:
                        new_seq_rewards = new_tmp_seq_rewards.reshape(1,len(rewards_per_episode))
                    else:
                        new_seq_rewards = new_tmp_seq_rewards.reshape(tmp_seq_rewards.shape[0]+1,tmp_seq_rewards.shape[1])
                    seq_rewards = new_seq_rewards.tolist()

                    tmp_seq_steps = np.array(seq_steps)
                    new_tmp_seq_steps = np.array(np.append(tmp_seq_steps.ravel(),np.array(steps_per_episode)))
                    if tmp_seq_steps.shape[0] == 0:
                        new_seq_steps = new_tmp_seq_steps.reshape(1,len(steps_per_episode))
                    else:
                        new_seq_steps = new_tmp_seq_steps.reshape(tmp_seq_steps.shape[0]+1,tmp_seq_steps.shape[1])
                    seq_steps = new_seq_steps.tolist()

                    q_tables[sim] = q_table

                    sim += 1
        tmp_rewards = np.array(all_rewards)
        new_tmp_rewards = np.array(np.append(tmp_rewards.ravel(),np.array(seq_rewards).ravel()))
        new_rewards = new_tmp_rewards.reshape(seq+1,sim,num_episodes)
        all_rewards = new_rewards.tolist()

        tmp_steps = np.array(all_steps)
        new_tmp_steps = np.array(np.append(tmp_steps.ravel(),np.array(seq_steps).ravel()))
        new_steps = new_tmp_steps.reshape(seq+1,sim,num_episodes)
        all_steps = new_steps.tolist()

        seq += 1

    avg_rewards, avg_steps = calc_avg(new_rewards, new_steps, num_sequences, num_sims, num_episodes)
    plot(q_tables, avg_rewards, avg_steps, learning_rate, discount_rate, exploration_rate)
    training_flag = False

    debug_q2 = input("See optimal policy?\nY/N?")
    debug_flag2 = str(debug_q2)

# if debug_flag2 == 'Y' or debug_flag2 == 'y':
if debug_flag == 2 or debug_flag2 == 'Y' or debug_flag2 == 'y':
    debug_q3 = input("Policy number?")
    policy = int(debug_q3)
    if debug_flag == 2: correct_path = load_path
    else: correct_path = save_path
    q_table_list, maze = extract_values(correct_path, policy)

    q_table = np.array(q_table_list)
    env.grid = np.array(maze)

    print(q_table)

    nb_success = 0
    # Display percentage successes
    for episode in range(100):
        print("Episode", episode)
        # initialize new episode params
        state = env.reset(1, episode)
        #print(env.grid)
        #print(env.grid)
        done = False
        for step in range(max_steps_per_episode):   
            # Show current state of environment on screen
            # Choose action with highest Q-value for current state (Greedy policy)     
            # Take new action
            action = np.argmax(q_table[state,:])  
            new_state, reward, done, info = env.step(action)
            
            if done:   
                nb_success += 1
                break

            # Set new state
            state = new_state

    # Let's check our success rate!
    print (f"Success rate = {nb_success}%")

    state = env.reset(1, episode)
    done = False

    for step in range(max_steps_per_episode):
        if debug_flag: all_grids.append(env.grid.copy())       
        # Show current state of environment on screen
        # Choose action with highest Q-value for current state (Greedy policy)     
        # Take new action
        action = np.argmax(q_table[state,:])  
        new_state, reward, done, info = env.step(action)
        
        if done:
            if debug_flag: all_grids.append(env.grid.copy())
            break

        # Set new state
        state = new_state

    if debug_flag:        
        for i in np.arange(len(all_grids)):
            PR = print_results(all_grids[i], env.grid.shape[0], env.grid.shape[1])
            PR.print_graph(episode, i)
            
            file_name = "plot%s-%s.png" %(str(episode),i)
            plt.savefig(os.path.join(save_path, file_name))
            plt.close()