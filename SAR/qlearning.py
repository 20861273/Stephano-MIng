from tkinter import LEFT
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import pandas as pd

from environment import Environment, HEIGHT, WIDTH, States, Direction, Point
from save_results import print_results

from datetime import datetime
import os
import time

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

    cum_avg_rewards = np.zeros(avg_rewards.shape)
    cum_avg_steps = np.zeros(avg_steps.shape)
    for i in range(len(avg_rewards)):
        r_numbers_series = pd.Series(avg_rewards[i])
        sim_avg_rewards = r_numbers_series.rolling(window=100).mean()
        s_numbers_series = pd.Series(avg_steps[i])
        sim_avg_steps = s_numbers_series.rolling(window=100).mean()
        
        cum_avg_rewards[i] = sim_avg_rewards
        cum_avg_steps[i] = sim_avg_steps

    return cum_avg_rewards.tolist(), cum_avg_steps.tolist()

def extract_values(policy_extraction, correct_path, policy, env):
    f = open(os.path.join(correct_path,"saved_data.txt"), "r")
    lines = f.readlines()
    WIDTH = 0
    HEIGHT = 0

    for line in lines:
        cur_num = ''
        cur_line = []
        if line[0:11] == "Grid shape:":
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
        if line[0:5] == "Goal:":
            for num in line:
                if num.isdigit():
                    cur_num += num
                else:
                    if cur_num == ',':
                        cur_line = []
                    elif cur_num != '': cur_line.append(int(cur_num))
                    cur_num = ''
            env.exit = Point(int(cur_line[0]),int(cur_line[1]))
    
    file_name = "maze.txt"
    maze =  np.loadtxt(os.path.join(correct_path, file_name))

    if policy_extraction:
        file_name = "policy" + str(policy) + ".txt"
        return np.loadtxt(os.path.join(correct_path, file_name)), maze
    else:
        return None, maze

class QLearning:
    def __init__(self):
        self.score = 0
        self.frame_iteration = 0

    def reset(self, env, generate):
        # init game state
        if generate:
            # Generates grid
            env.grid = env.generate_grid()
            env.prev_pos = env.starting_pos
            env.pos = env.prev_pos
        else:
            # # Code that spawns target a certain distance away from agent
            # distance_to = 0
            # while distance_to < env.grid.shape[0]*0.2:
            #     indices = np.argwhere(env.grid == States.UNEXP.value)
            #     np.random.shuffle(indices)
            #     env.goal = Point(indices[0,1], indices[0,0])
            #     distance_to = math.sqrt((env.starting_pos.x - env.goal.x)**2 +
            #                     (env.starting_pos.y - env.goal.y)**2)


            # Constant goal, varied starting pos
            # # Setup goal
            # env.grid[env.goal.y, env.goal.x] = States.GOAL.value

            # visited = np.argwhere(env.grid == States.EXP.value)
            # for i in visited:
            #     env.grid[i[0], i[1]] = States.UNEXP.value

            # # Setup robot starting position
            # env.grid[env.pos.y, env.pos.x] = States.UNEXP.value

            # distance_to = 0
            # while distance_to < env.grid.shape[0]*0.6:
            #     indices = np.argwhere(env.grid == States.UNEXP.value)
            #     np.random.shuffle(indices)
            #     env.starting_pos = Point(indices[0,1], indices[0,0])
            #     distance_to = math.sqrt((env.starting_pos.x - env.goal.x)**2 +
            #                     (env.starting_pos.y - env.goal.y)**2)

            # env.pos = env.starting_pos
            # env.prev_pos = env.pos
            # env.grid[env.pos.y, env.pos.x] = States.ROBOT.value


            # # Constant starting pos, varied goal
            # # Setup agent
            # env.grid[env.pos.y, env.pos.x] = States.UNEXP.value
            # env.pos = env.starting_pos
            # env.prev_pos = env.pos
            # env.grid[env.pos.y, env.pos.x] = States.ROBOT.value

            # visited = np.argwhere(env.grid == States.EXP.value)
            # for i in visited:
            #     env.grid[i[0], i[1]] = States.UNEXP.value

            # # Setup goal
            # env.grid[env.goal.y, env.goal.x] = States.UNEXP.value
            # indices = np.argwhere(env.grid == States.UNEXP.value)
            # np.random.shuffle(indices)
            # env.goal = Point(indices[0,1], indices[0,0])
            # env.grid[env.goal.y, env.goal.x] = States.GOAL.value

            # Varied starting pos and goal
            visited = np.argwhere(env.grid == States.EXP.value)
            for i in visited:
                env.grid[i[0], i[1]] = States.UNEXP.value

            # Setup agent
            env.grid[env.pos.y, env.pos.x] = States.UNEXP.value
            indices = np.argwhere(env.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            env.starting_pos = Point(indices[0,1], indices[0,0])
            env.pos = env.starting_pos
            env.prev_pos = env.pos
            env.grid[env.pos.y, env.pos.x] = States.ROBOT.value

            # Setup goal
            env.grid[env.goal.y, env.goal.x] = States.UNEXP.value
            indices = np.argwhere(env.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            env.goal = Point(indices[0,1], indices[0,0])
            env.grid[env.goal.y, env.goal.x] = States.GOAL.value
        
        env.direction = (Direction.RIGHT).value
                
        self.score = 0
        self.frame_iteration = 0

        state = self.get_state(env)

        return state
        
    def step(self, env, action):
        #print(self.pos, self.prev_pos, action)
        #print(self.grid,"\n")
        self.frame_iteration += 1
        # 2. Do action
        self._move(env, action) # update the robot
            
        # 3. Update score and get state
        self.score = 0
        self.score -= 0.1
        game_over = False

        state = self.get_state(env)
        
        reward = self.score

        # 4. Update environment
        self._update_env(env)

        # 5. Check exit condition
        if env.pos == env.goal:
            self.score += 100
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
            return True
        
        return False
        
    def _update_env(self, env):
        if self.frame_iteration == 0:
            # Update robot position(s) on grid
            env.grid[env.pos.y,env.pos.x] = States.ROBOT.value
        else:
            # Update robot position(s) on grid
            env.grid[env.prev_pos.y,env.prev_pos.x] = States.EXP.value
            env.grid[env.pos.y,env.pos.x] = States.ROBOT.value
            

    def _move(self, env, action):
        if action == (Direction.LEFT).value:
            env.direction = action
        elif action == (Direction.RIGHT).value:
            env.direction = action
        elif action == (Direction.UP).value:
            env.direction = action
        elif action == (Direction.DOWN).value:
            env.direction = action

        x = env.pos.x
        y = env.pos.y
        if env.direction == (Direction.RIGHT).value:
            x += 1
        elif env.direction == (Direction.LEFT).value:
            x -= 1
        elif env.direction == (Direction.DOWN).value:
            y += 1
        elif env.direction == (Direction.UP).value:
            y -= 1

        if self._is_collision(env, Point(x,y)):
            env.pos = env.pos
            env.prev_pos = env.pos
        else:
            env.prev_pos = env.pos
            env.pos = Point(x,y)

    def run_qlearning(self, mode):
        # Creating The Environment
        env = Environment()
        QL = QLearning()

        # Creating The Q-Table
        action_space_size = len(Direction)
        state_space_size = env.grid.shape[1] * env.grid.shape[0]
        #print(state_space_size, action_space_size)

        q_table = np.zeros((state_space_size, action_space_size))

        # Initializing Q-Learning Parameters
        num_episodes = 10000
        max_steps_per_episode = 200
        num_sequences = 1

        learning_rate = np.array([0.01])
        discount_rate = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) # 0.9
        # Optimal hyperparameters:
        # 5x5: lr=0.02, dr=0.7
        # 8x8: lr= ,dr=
        pos_reward = env.grid.shape[0]*env.grid.shape[1]*5

        exploration_rate = np.array([0.01], dtype=np.float32) # 0.01
        max_exploration_rate = np.array([0.01], dtype=np.float32)
        min_exploration_rate = np.array([0.01], dtype=np.float32)
        exploration_decay_rate = np.array([0.01], dtype=np.float32)

        generate = True
        policy_extraction = True
        if mode == 2:
            generate = False
        elif mode == 3:
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

        print("\n\n# Training sessions: ", num_sequences, "\n# Simulations per training session: ", num_sims, "\n# Episodes per simulation: ", num_episodes)
        print("Hyperparameters:\nLearning rate (α): ", learning_rate, "\nDiscount rate (γ): ", discount_rate, "\nExploration rate (ϵ): ", exploration_rate, "\nExploration decay rate: ", exploration_decay_rate)

        if mode != 1:
            PATH = os.getcwd()
            PATH = os.path.join(PATH, 'SAR')
            PATH = os.path.join(PATH, 'Results')
            PATH = os.path.join(PATH, 'QLearning')
            load_path = os.path.join(PATH, 'Saved_data')
            if not os.path.exists(load_path): os.makedirs(load_path)
            date_and_time = datetime.now()
            save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
            if not os.path.exists(save_path): os.makedirs(save_path)
        if mode == 3:
            _, env.grid = extract_values(policy_extraction, load_path, None, env)
        if mode != 2:
            PATH = os.getcwd()
            PATH = os.path.join(PATH, 'SAR')
            PATH = os.path.join(PATH, 'Results')
            PATH = os.path.join(PATH, 'QLearning')
            load_path = os.path.join(PATH, 'Saved_data')
            if not os.path.exists(load_path): os.makedirs(load_path)
            date_and_time = datetime.now()
            save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
            if not os.path.exists(save_path): os.makedirs(save_path)

        if mode == 1 or mode == 3:
            # Training loop
            training_time = time.time()
            generate = False
            state = QL.reset(env, generate)
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
                                if episode % 1000 == 0: print("Episode: ", episode)
                                
                                # Initialize new episode params
                                state = QL.reset(env, generate)
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
                                    new_state, reward, done, info = QL.step(env, action)

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
                                if mode:
                                    if not done:
                                        rewards_per_episode.append(rewards_current_episode)
                                        steps_per_episode.append(step+1)

                            if mode:
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
            training_time = time.time() - training_time
            print("Time to train policy: ", training_time)

            results = print_results(env.grid, HEIGHT, WIDTH)
            QL.reset(env, generate)
            results.plot(q_tables, avg_rewards, avg_steps, learning_rate, discount_rate, exploration_rate, load_path, env)

            test_tab = np.empty(shape=env.grid.shape).tolist()
            for s in range(env.grid.shape[0]*env.grid.shape[1]):
                a = np.argmax(q_table[s,:])
                if a == Direction.RIGHT.value: test_tab[s] = '→'
                elif a == Direction.LEFT.value: test_tab[s] = '←'
                elif a == Direction.UP.value: test_tab[s] = '↑'
                elif a == Direction.DOWN.value: test_tab[s] = '↓'

            print(test_tab)

            debug_q2 = input("See optimal policy?\nY/N?")
            debug_flag2 = str(debug_q2)

        if mode == 2 or debug_flag2 == 'Y' or debug_flag2 == 'y':
            policy_extraction = True
            debug_q3 = input("Policy number?")
            policy = int(debug_q3)
            if mode == 2: correct_path = load_path
            else: correct_path = save_path
            q_table_list, maze = extract_values(policy_extraction, load_path, policy, env)

            q_table = np.array(q_table_list)
            env.grid = np.array(maze)

            nb_success = 0
            # Display percentage successes
            for episode in range(1000):
                if episode % 100 == 0: print("Episode: ", episode)
                # initialize new episode params
                state = QL.reset(env, generate)
                #print(env.grid)
                #print(env.grid)
                done = False
                for step in range(max_steps_per_episode):   
                    # Show current state of environment on screen
                    # Choose action with highest Q-value for current state (Greedy policy)     
                    # Take new action
                    action = np.argmax(q_table[state,:])
                    new_state, reward, done, info = QL.step(env, action)
                    
                    if done:   
                        nb_success += 1
                        break

                    # Set new state
                    state = new_state

            # Let's check our success rate!
            print (f"Success rate = {nb_success/10}%")

            state = QL.reset(env, generate)
            env.grid[env.pos.y, env.pos.x] = States.UNEXP.value
            env.prev_pos = env.starting_pos
            env.pos = env.prev_pos
            env.grid[env.pos.y, env.pos.x] = States.ROBOT.value
            done = False

            for step in range(max_steps_per_episode):
                if mode: all_grids.append(env.grid.copy())       
                # Show current state of environment on screen
                # Choose action with highest Q-value for current state (Greedy policy)     
                # Take new action
                action = np.argmax(q_table[state,:]) 
                new_state, reward, done, info = QL.step(env, action)
                
                if done:
                    if mode: all_grids.append(env.grid.copy())
                    break

                # Set new state
                state = new_state

            if mode:        
                for i in np.arange(len(all_grids)):
                    PR = print_results(all_grids[i], env.grid.shape[0], env.grid.shape[1])
                    PR.print_graph(episode, i)
                    
                    file_name = "plot%s-%s.png" %(str(episode),i)
                    plt.savefig(os.path.join(save_path, file_name))
                    plt.close()