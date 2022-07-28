import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from maze import Maze, HEIGHT, WIDTH, States, Direction, Point

from datetime import datetime
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

        plt_title = "Q-learning Results: Episode %s, step %s" %(str(episode), str(step)) 
        plt.title(plt_title)

def plot(q_tables, rewards, steps, learning_rate, discount_rate, exploration_rate, save_path, env):
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

    f.write(str("Maze shape: %s\nStarting position: %s\nExit: %s\nMaze:\n%s" %(str(env.grid.shape), str(env.starting_pos), str(env.exit), str(env.grid))))
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

def extract_values(policy_extraction, correct_path, policy, env):
    maze = []
    f = open(os.path.join(correct_path,"saved_data.txt"), "r")
    lines = f.readlines()
    maze_flag = False
    WIDTH = 0
    HEIGHT = 0

    for line in lines:
        cur_num = ''
        cur_line = []
        if line[0:11] == "Maze shape:":
            for num in line:
                if num.isdigit():
                    cur_num += num
                else:
                    if cur_num != '': cur_line.append(int(cur_num))
                    cur_num = ''
            WIDTH = int(cur_line[1])
            HEIGHT = int(cur_line[0])

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

    if policy_extraction:
        file_name = "policy" + str(policy) + ".txt"
        return np.loadtxt(os.path.join(correct_path, file_name)), np.array(maze)
    else:
        return None, np.array(maze)

class QLearning:
    def __init__(self):
        self.score = 0
        self.frame_iteration = 0

    def reset(self, env, generate):
        # print("Unexplored block = 0\n",
        #     "Obstacle = 1\n"
        #     "Robot = 2\n"
        #     "Exit = 3\n"
        #     "Explored block = 4")
        # init game state
        if generate:
            # Generates grid
            env.grid = env.generate_grid()
            env.prev_pos = env.starting_pos
            env.pos = env.prev_pos
        else:
            # Setup robot starting position
            env.grid[env.pos.y, env.pos.x] = States.UNEXP.value
            env.pos = env.starting_pos
            env.prev_pos = env.pos
            env.grid[env.pos.y, env.pos.x] = States.ROBOT.value

            # Setup maze exit
            env.grid[env.exit.y, env.exit.x] = States.EXIT.value
        
        #print(self.grid)
        env.direction = (Direction.RIGHT).value
                
        self.score = 0
        self.frame_iteration = 0

        state = self.get_state(env)

        return state
        
    def step(self, env, action):
        #print(self.pos, self.prev_pos, action)
        #print(self.grid,"\n")
        self.frame_iteration += 1
        # 2. move
        self._move(env, action) # update the robot
            
        # 3. check if game over
        reward = 0
        self.score -= 0.1
        game_over = False

        state = self.get_state(env)
        
        reward = self.score

        # 4. update maze
        self._update_maze(env)

        # 5. reached exit or just move
        if env.pos == env.exit:
            #self.score += 1
            reward = self.score
            game_over = True
            return state, reward, game_over, self.score
        
        # 6. return game over and score
        return state, reward, game_over, self.score

    def get_state(self, env):
        return env.pos.x*env.grid.shape[1] + env.pos.y
    
    def _is_collision(self, env, pt=None):
        if pt is None:
            pt = env.pos
        # hits boundary
        obstacles = np.argwhere(env.grid == 1)
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            return True
        elif pt.y < 0 or pt.y > env.grid.shape[0]-1 or pt.x < 0 or pt.x > env.grid.shape[1]-1:
            return True
        
        return False

    def _is_explored(self, env, pt=None):
        if pt is None:
            pt = env.pos
        # hits boundary
        explored = np.argwhere(env.grid == States.EXP.value)
        if any(np.equal(explored,np.array([env.pos.y,env.pos.x])).all(1)):
            #print(self.pos)
            return True
        
        return False
        
    def _update_maze(self, env):
        if self.frame_iteration == 0:
            # Update robot position(s) on grid
            env.grid[env.pos.y,env.pos.x] = States.ROBOT.value
        else:
            # Update robot position(s) on grid
            env.grid[env.prev_pos.y,env.prev_pos.x] = States.UNEXP.value
            env.grid[env.pos.y,env.pos.x] = States.ROBOT.value
            

    def _move(self, env, action):
        if action == (Direction.LEFT).value:
            env.direction = action
            #print(action, (Direction.LEFT).value, self.direction)
        elif action == (Direction.RIGHT).value:
            env.direction = action
            #print(action, (Direction.RIGHT).value, self.direction)
        elif action == (Direction.UP).value:
            env.direction = action
            #print(action, (Direction.UP).value, self.direction)
        elif action == (Direction.DOWN).value:
            env.direction = action
            #print(action, (Direction.DOWN).value, self.direction)

        x = env.pos.x
        y = env.pos.y
        if env.direction == (Direction.RIGHT).value:
            x += 1
            #print("RIGHT")
        elif env.direction == (Direction.LEFT).value:
            x -= 1
            #print("LEFT")
        elif env.direction == (Direction.DOWN).value:
            y += 1
            #print("DOWN")
        elif env.direction == (Direction.UP).value:
            y -= 1
            #print("UP")

        if self._is_collision(env, Point(x,y)):
            env.pos = env.pos
            env.prev_pos = env.pos
        else:
            env.prev_pos = env.pos
            env.pos = Point(x,y)
            #print(action, self.prev_pos, self.pos)

        #print(action, self.prev_pos, self.pos)

def run_qlearning():
    # Creating The Environment
    env = Maze()
    envM = QLearning()

    # Creating The Q-Table
    action_space_size = len(Direction)
    state_space_size = env.grid.shape[1] * env.grid.shape[0]
    #print(state_space_size, action_space_size)

    q_table = np.zeros((state_space_size, action_space_size))

    # Initializing Q-Learning Parameters
    num_episodes = 1000
    max_steps_per_episode = 10000
    num_sequences = 1

    learning_rate = np.array([0.1]) # 0.01
    discount_rate = np.array([0.99]) # 0.9

    exploration_rate = np.array([0.01], dtype=np.float32) # 0.01
    max_exploration_rate = np.array([0.01], dtype=np.float32)
    min_exploration_rate = np.array([0.01], dtype=np.float32)
    exploration_decay_rate = np.array([0.01], dtype=np.float32)

    # Debug menu
    debug_q = input("\n\nDefault training mode: 1\nGreedy policy: 2\nGenerated map: 3\nSelect mode: ")
    # debug_q = input("Default training mode: 0\nDebug training mode: 1\nSelect mode: ")
    debug_flag = int(debug_q)

    generate = True
    policy_extraction = True
    if debug_flag == 2:
        generate = False
    elif debug_flag == 3:
        generate = False
        policy_extraction = False

    

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

    if debug_flag != 1:
        PATH = os.getcwd()
        PATH = os.path.join(PATH, 'Results')
        PATH = os.path.join(PATH, 'QLearning')
        load_path = os.path.join(PATH, 'saved_data')
        if not os.path.exists(load_path): os.makedirs(load_path)
    if debug_flag == 3:
        _, env.grid = extract_values(policy_extraction, load_path, None, env)
    if debug_flag != 2:
        PATH = os.getcwd()
        PATH = os.path.join(PATH, 'Results')
        PATH = os.path.join(PATH, 'QLearning')
        date_and_time = datetime.now()
        save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
        if not os.path.exists(save_path): os.makedirs(save_path)

    if debug_flag == 1 or debug_flag == 3:
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
                            state = envM.reset(env, generate)
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
                                new_state, reward, done, info = envM.step(env, action)

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
                            if debug_flag:
                                if not done:
                                    rewards_per_episode.append(rewards_current_episode)
                                    steps_per_episode.append(step+1)

                        if debug_flag:
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
        plot(q_tables, avg_rewards, avg_steps, learning_rate, discount_rate, exploration_rate, save_path, env)

        debug_q2 = input("See optimal policy?\nY/N?")
        debug_flag2 = str(debug_q2)

    if debug_flag == 2 or debug_flag2 == 'Y' or debug_flag2 == 'y':
        policy_extraction = True
        debug_q3 = input("Policy number?")
        policy = int(debug_q3)
        if debug_flag == 2: correct_path = load_path
        else: correct_path = save_path
        q_table_list, maze = extract_values(policy_extraction, correct_path, policy, env)

        q_table = np.array(q_table_list)
        env.grid = np.array(maze)

        nb_success = 0
        # Display percentage successes
        for episode in range(100):
            print("Episode", episode)
            # initialize new episode params
            state = envM.reset(env, generate)
            #print(env.grid)
            #print(env.grid)
            done = False
            for step in range(max_steps_per_episode):   
                # Show current state of environment on screen
                # Choose action with highest Q-value for current state (Greedy policy)     
                # Take new action
                action = np.argmax(q_table[state,:])  
                new_state, reward, done, info = envM.step(env, action)
                
                if done:   
                    nb_success += 1
                    break

                # Set new state
                state = new_state

        # Let's check our success rate!
        print (f"Success rate = {nb_success}%")

        state = envM.reset(env, generate)
        done = False

        for step in range(max_steps_per_episode):
            if debug_flag: all_grids.append(env.grid.copy())       
            # Show current state of environment on screen
            # Choose action with highest Q-value for current state (Greedy policy)     
            # Take new action
            action = np.argmax(q_table[state,:])  
            new_state, reward, done, info = envM.step(env, action)
            
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