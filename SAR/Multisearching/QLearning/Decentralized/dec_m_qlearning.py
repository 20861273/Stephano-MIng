from pickletools import uint8
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import winsound

from dec_m_environment import Environment, HEIGHT, WIDTH, States, Direction, Point
from dec_save_results import print_results

from datetime import datetime
import os
import time
import shutil

class QLearning:
    def __init__(self,env):
        self.frame_iteration = 0
        self.nr = 2
        self.score = [0]*self.nr
        self.starting_pos = env.starting_pos
        self.pos = env.pos
        self.prev_pos = env.prev_pos
        self.goal = env.goal

    def run_qlearning(self, mode):
        # Creating The Environment
        env = Environment(self.nr)

        # Creating The Q-Table
        action_space_size = len(Direction)
        state_space_size = (env.grid.shape[1] * env.grid.shape[0])**2
        #print(state_space_size, action_space_size)

        q_table = np.zeros((self.nr, state_space_size, action_space_size))

        # Initializing Q-Learning Parameters
        num_episodes = 5000000
        max_steps_per_episode = 200
        epochs = 1

        learning_rate = np.array([0.01])
        discount_rate = np.array([0.9])

        pos_reward = env.grid.shape[0]*env.grid.shape[1]
        # pos_reward = 1
        n_tests = 50
        interval = 5000

        exploration_rate = np.array([0.1, 1], dtype=np.float32)
        max_exploration_rate = np.array([0.1, 1], dtype=np.float32)
        min_exploration_rate = np.array([0.1, 0.1], dtype=np.float32)
        exploration_decay_rate = np.array([0.1, 0.001], dtype=np.float32)

        generate = True
        policy_extraction = True
        if mode == 2:
            generate = False
        elif mode == 3:
            generate = False
            policy_extraction = False

        # Result parameters
        experiments = len(learning_rate) * len(discount_rate) * len(exploration_rate)
        exp = 0
        epoch_cnt = 0
        rewards_per_episode = []
        seq_rewards = []
        avg_rewards = []
        all_rewards = []
        all_grids = []
        action = np.empty((self.nr,), dtype=np.int8)
        q_tables = np.zeros((experiments, self.nr, state_space_size, action_space_size))

        print("\n# Epochs: ", epochs, "\n# Experiments: ", experiments, "\n# Episodes: ", num_episodes)
        print("\nHyperparameters:\nLearning rate (α): ", learning_rate, "\nDiscount rate (γ): ", discount_rate, "\nExploration rate (ϵ): ", exploration_rate, "\nExploration decay rate: ", exploration_decay_rate, "\n")

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
            _, env.grid.shape = extract_values(policy_extraction, load_path, None, env)
        if mode != 2:
            PATH = os.getcwd()
            PATH = os.path.join(PATH, 'SAR')
            PATH = os.path.join(PATH, 'Results')
            PATH = os.path.join(PATH, 'QLearning')
            PATH = os.path.join(PATH, 'Decentralized')
            load_path = os.path.join(PATH, 'Saved_data')
            if not os.path.exists(load_path): os.makedirs(load_path)
            date_and_time = datetime.now()
            save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
            if not os.path.exists(save_path): os.makedirs(save_path)

        if mode == 1 or mode == 3:
            # Training loop
            training_time = time.time()
            generate = False
            for seq_i in range(0, epochs):
                print("Epoch: ", epoch_cnt)
                exp_rewards = []
                seq_steps = []
                exp = 0
                for lr_i in np.arange(len(learning_rate)):
                    for dr_i in np.arange(len(discount_rate)):
                        for er_i in np.arange(len(exploration_rate)):
                            print("Experiment: ", exp)
                            # Reinitialize some variables
                            ep_exploration_rate = np.copy(exploration_rate)
                            q_table = np.zeros((self.nr, state_space_size, action_space_size))

                            # Training Loop
                            rewards_per_episode = []                

                            # Q-learning algorithm
                            for episode in range(num_episodes):
                                if episode % 10000 == 0:
                                    print("Episode: ", episode)
                                
                                # Initialize new episode params
                                state = self.reset(env, generate)
                                done = False
                                rewards_current_episode = [0]*self.nr
                                reward = 0

                                for step in range(1, max_steps_per_episode+1):
                                    for i in range(0, self.nr):
                                        # Exploration-exploitation trade-off
                                        exploration_rate_threshold = random.uniform(0, 1)
                                        if exploration_rate_threshold > ep_exploration_rate[er_i]:
                                            action[i] = np.argmax(q_table[i, state[i],:])
                                        else:
                                            action_space_actual_size = action_space_size-1
                                            action[i] = random.randint(0, action_space_actual_size)

                                    # Take new action
                                    new_state, reward, done, info = self.step(env, action, pos_reward, step)

                                    # Update Q-table
                                    for i in range(0, self.nr):
                                        q_table[i, state[i], action] = q_table[i, state[i], action] * (1 - learning_rate[lr_i]) + \
                                            learning_rate[lr_i] * (reward[i] + discount_rate[dr_i] * np.max(q_table[i, new_state[i], :]))

                                    # Set new state
                                    state = new_state
                                    
                                    # Add new reward  
                                    rewards_current_episode[0] += reward[0]
                                    rewards_current_episode[1] += reward[1]


                                    if done == True:
                                        rewards_per_episode.append(rewards_current_episode)
                                        break

                                    # Exploration rate decay 
                                    ep_exploration_rate[er_i] = min_exploration_rate[er_i] + \
                                        (max_exploration_rate[er_i] - min_exploration_rate[er_i]) * np.exp(-exploration_decay_rate[er_i]*episode)

                                # Add current episode reward to total rewards list
                                if mode:
                                    if not done:
                                        rewards_per_episode.append(rewards_current_episode)

                            if mode:
                                # Adds epoch rewards to training session rewards variable
                                # rewards_per_episode: (num_episodes, self.nr) ; i.e,  [[agent0_reward0, agent1_reward0], [agent0_reward1, agent1_reward1], ...]
                                # new_tmp_exp_rewards: (2*num_episodes,)
                                # new_exp_rewards: (num_sequences, self.nr, num_episodes)
                                rewards_per_episode = rewards_per_episode[::interval]
                                tmp_exp_rewards = np.array(exp_rewards)
                                new_tmp_exp_rewards = np.array(np.append(tmp_exp_rewards.ravel(),np.array(rewards_per_episode).T))
                                if tmp_exp_rewards.shape[0] == 0:
                                    new_exp_rewards = new_tmp_exp_rewards.reshape(1,self.nr, len(rewards_per_episode))
                                else:
                                    new_exp_rewards = new_tmp_exp_rewards.reshape(new_exp_rewards.shape[0]+1, self.nr, new_exp_rewards.shape[2])
                                exp_rewards = new_exp_rewards.tolist()
                                
                                q_tables[exp] = q_table

                            exp += 1
                tmp_rewards = np.array(all_rewards)
                new_tmp_rewards = np.array(np.append(tmp_rewards.ravel(),np.array(exp_rewards).ravel()))
                new_rewards = new_tmp_rewards.reshape(epoch_cnt+1,exp,self.nr,int(num_episodes/interval))
                all_rewards = new_rewards.tolist()

                epoch_cnt += 1
            
            avg_rewards = calc_avg(new_rewards, epochs, experiments, self.nr)
            training_time = time.time() - training_time
            print("Time to train policy: %sm %ss" %(divmod(training_time, 60)))

            results = print_results(env.grid, HEIGHT, WIDTH)
            self.reset(env, generate)

            for j in range(0, 2):
                trajs = []
                for i in range(0, q_tables.shape[0]):
                    print("\nTrajectories of policy %s:" %(i))
                    test_tab = [None] * (env.grid.shape[1]*env.grid.shape[0])
                    for s in range(env.grid.shape[0]*env.grid.shape[1]):
                        a = np.argmax(q_tables[i,j,s,:])
                        if a == Direction.RIGHT.value:
                            test_tab[s] = ">"
                        elif a == Direction.LEFT.value: 
                            test_tab[s] = "<"
                        elif a == Direction.UP.value: 
                            test_tab[s] = "^"
                        elif a == Direction.DOWN.value: 
                            test_tab[s] = "v"
                
                    trajs.append(np.reshape(test_tab, (env.grid.shape[1], env.grid.shape[0])).T)
                    print(np.reshape(test_tab, (env.grid.shape[1], env.grid.shape[0])).T)

            results.plot_and_save(q_tables, avg_rewards, learning_rate, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, save_path, env, training_time)

            for i in range(0, q_tables.shape[0]):
                print("\nTesting policy %s:" % (i))
                nb_success = [0]*n_tests
                zeros0 = [0]*q_tables.shape[0]
                zeros1 = [0]*q_tables.shape[0]
                # Display percentage successes
                for j in range(0, n_tests):
                    if j % 10 == 0: print("Testing...", j)
                    zeros0[i] = 0
                    zeros1[i] = 0
                    for state0 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                        for state1 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                            # if episode % 1000 == 0: print("Episode: ", episode)
                            # initialize new episode params
                            _ = self.reset(env, True)
                            state = [(state0*env.grid.shape[0]*env.grid.shape[1] + state1),
                                (state1*env.grid.shape[0]*env.grid.shape[1] + state0)]
                            tmp_zeros0 = [a for a in q_tables[i, 0, state[0]] if a == 0]
                            zeros0[i] += len(tmp_zeros0)
                            tmp_zeros1 = [a for a in q_tables[i, 1, state[0]] if a == 0]
                            zeros1[i] += len(tmp_zeros1)
                            
                            for step in range(1, max_steps_per_episode+1):   
                                # Show current state of environment on screen
                                # Choose action with highest Q-value for current state (Greedy policy)     
                                # Take new action
                                action[0] = np.argmax(q_tables[i, 0, state[0],:])
                                action[1] = np.argmax(q_tables[i, 1, state[1],:])
                                new_state, reward, done, info = self.step(env, action, pos_reward, step)
                                
                                if done:   
                                    nb_success[j] += 1
                                    break

                                # Set new state
                                state = new_state

                # Let's check our success rate!
                success = sum(nb_success)/n_tests

                # Let's check our success rate!
                print ("Success rate of policy %s: %s / %s * 100 = %s %%" % (i, success, state_space_size, success/state_space_size*100))
                print("Agent 0: %s\nAgent 1: %s\nOut of: %s" %(str(zeros0[i]), str(zeros1[i]), str(state_space_size*action_space_size**2)))

            winsound.Beep(1000, 500)

            debug_q2 = input("See optimal policy?\nY/N?")
            debug_flag2 = str(debug_q2)

        if mode == 2 or debug_flag2 == 'Y' or debug_flag2 == 'y':
            policy_extraction = True
            debug_q3 = input("Policy number?")
            policy = int(debug_q3)
            if mode == 2: correct_path = load_path
            else: correct_path = save_path
            q_table_list0, q_table_list1, env_shape = extract_values(policy_extraction, load_path, policy, env)

            q_table = np.array([[q_table_list0], [q_table_list1]])
            env.grid = np.zeros((int(env_shape[0]), int(env_shape[1])))

            state = self.reset(env, generate)
            done = False

            nb_success = [0]*n_tests
            zeros0 = [0]*n_tests
            zeros1 = [0]*n_tests
            # Display percentage successes
            for j in range(0, n_tests):
                if j % 10 == 0: print("Testing...", j)
                for state0 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                    for state1 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                        # if episode % 1000 == 0: print("Episode: ", episode)
                        # initialize new episode params
                        _ = self.reset(env, True)
                        tmp_zeros0 = [a for a in q_table[0, state0] if a == 0]
                        zeros0[j] += len(tmp_zeros0)
                        tmp_zeros1 = [a for a in q_table[1, state0] if a == 0]
                        zeros1[j] += len(tmp_zeros1)
                        # print(state0, state1)
                        #print(env.grid)
                        #print(env.grid)
                        for step in range(1, max_steps_per_episode+1):   
                            # Show current state of environment on screen
                            # Choose action with highest Q-value for current state (Greedy policy)     
                            # Take new action
                            action[0] = np.argmax(q_table[0, state0,:])
                            action[1] = np.argmax(q_table[1, state1,:])
                            new_state, reward, done, info = self.step(env, action, pos_reward, step)
                            
                            if done:   
                                nb_success[j] += 1
                                break

                            # Set new state
                            state = new_state

            # Let's check our success rate!
            success = sum(nb_success)/n_tests

            # Let's check our success rate!
            print ("Success rate of policy %s: %s / %s * 100 = %s %%" % (i, success, state_space_size, success/state_space_size*100))
            print("Agent 0: %s\nAgent 1: %s\nOut of: %s" %(str(zeros0[i]), str(zeros1[i]), str(state_space_size*action_space_size)))

            if mode == 3:
                for step in range(1, max_steps_per_episode+1):
                    if mode: all_grids.append(env.grid.copy())       
                    # Show current state of environment on screen
                    # Choose action with highest Q-value for current state (Greedy policy)     
                    for i in range(0, self.nr):  
                        # Take new action
                        action[i] = np.argmax(q_table[i, 0, state[i],:])
                    new_state, reward, done, info = self.step(env, action, pos_reward, step)
                    
                    if done:
                        print("done", self.pos, env.goal)
                        if mode: all_grids.append(env.grid.copy())
                        break

                    # Set new state
                    state = new_state

                    
                for i in np.arange(len(all_grids)):
                    PR = print_results(all_grids[i], env.grid.shape[0], env.grid.shape[1])
                    PR.print_graph(i)
                    
                    file_name = "plot-%s.png" %(i)
                    plt.savefig(os.path.join(save_path, file_name))
                    plt.close()


    def reset(self, env, generate):
        # init game state
        if generate:
            # Generates grid
            env.grid = env.generate_grid(self.nr)
            self.prev_pos = self.starting_pos
            self.pos = self.prev_pos

            for i in range(0, self.nr):
                env.grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            self.goal = env.goal
        else:
            # # Code that spawns target a certain distance away from agent
            # distance_to = 0
            # while distance_to < env.grid.shape[0]*0.2:
            #     indices = np.argwhere(env.grid == States.UNEXP.value)
            #     np.random.shuffle(indices)
            #     env.goal = Point(indices[0,1], indices[0,0])
            #     distance_to = math.sqrt((env.starting_pos.x - env.goal.x)**2 +
            #                     (env.starting_pos.y - env.goal.y)**2)

            # # Varied starting pos and goal
            # # Clear all visited blocks
            # visited = np.argwhere(env.grid == States.EXP.value)
            # for i in visited:
            #     env.grid[i[0], i[1]] = States.UNEXP.value

            # # Setup agent
            # # Clear all robot blocks
            # robots = np.argwhere(env.grid == States.ROBOT.value)
            # for i in robots:
            #     env.grid[i[0], i[1]] = States.UNEXP.value
            # # Set robot(s) start position
            # self.starting_pos = [0]*self.nr
            # indices = np.argwhere(env.grid == States.UNEXP.value)
            # np.random.shuffle(indices)
            # for i in range(0, self.nr):
            #     self.starting_pos[i] = (Point(indices[i,1], indices[i,0]))
            #     distance_to = 0
            #     next_i = i
            #     if not i == 0:
            #         while distance_to < env.grid.shape[0]*0.7:
            #             self.starting_pos[i] = Point(indices[next_i,1], indices[next_i,0])
            #             distance_to = math.sqrt((self.starting_pos[i].x - self.starting_pos[i-1].x)**2 +
            #                             (self.starting_pos[i].y - self.starting_pos[i-1].y)**2)
            #             next_i += 1

            #     env.grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
            # self.pos = self.starting_pos

            # # Set goal position
            # goal = np.argwhere(env.grid == States.GOAL.value)
            # for i in goal:
            #     env.grid[i[0], i[1]] = States.UNEXP.value
            # indices = np.argwhere(env.grid == States.UNEXP.value)
            # np.random.shuffle(indices)
            # self.goal = Point(indices[0,1], indices[0,0])
        
            # env.grid[self.goal.y, self.goal.x] = States.GOAL.value

            # Varied starting pos and constant goal
            # Set goal position        
            env.grid[self.goal.y, self.goal.x] = States.GOAL.value
            # Clear all visited blocks
            visited = np.argwhere(env.grid == States.EXP.value)
            for i in visited:
                env.grid[i[0], i[1]] = States.UNEXP.value

            # Setup agent
            # Clear all robot blocks
            robots = np.argwhere(env.grid == States.ROBOT.value)
            for i in robots:
                env.grid[i[0], i[1]] = States.UNEXP.value
            # Set robot(s) start position
            # self.starting_pos = [0]*self.nr
            # indices = np.argwhere(env.grid == States.UNEXP.value)
            # np.random.shuffle(indices)
            # i = 0
            # self.starting_pos[0] = Point(indices[i,1], indices[i,0])
            # self.starting_pos[1] = self.starting_pos[0]

            # while not self.starting_pos[0].y < env.grid.shape[0]/2:
            #     self.starting_pos[0] = Point(indices[i,1], indices[i,0])
            #     i += 1
            # while not self.starting_pos[1].y >= env.grid.shape[0]/2:
            #     self.starting_pos[1] = Point(indices[i,1], indices[i,0])
            #     i += 1
            
            self.starting_pos = [0]*self.nr
        indices = np.argwhere(env.grid == States.UNEXP.value)
        np.random.shuffle(indices)
        
        self.starting_pos[0] = Point(indices[0,0], indices[0,1])
        self.starting_pos[1] = self.starting_pos[0]
        while not self.starting_pos[1] == self.starting_pos[0]:
            np.random.shuffle(indices)
            self.starting_pos[1] = Point(indices[0,0], indices[0,1])
        
        for i in range(0, self.nr):
            env.grid[self.starting_pos[i].y, self.starting_pos[i].x] = States.ROBOT.value
            
            self.pos = self.starting_pos            
        
        self.direction = [(Direction.RIGHT).value]*self.nr
                
        self.score = [0]*self.nr
        self.frame_iteration = 0

        state = self.get_state(env)

        return state
        
    def step(self, env, action, pos_reward, step):
        #print(self.pos, self.prev_pos, action)
        #print(self.grid,"\n")
        self.frame_iteration += 1
        self.p_reward = pos_reward
        self.score = [0]*self.nr
        # 2. Do action
        self._move(env, action) # update the robot
            
        # 3. Update score and get state
        self.score = [i-0.1 for i in self.score]
        game_over = False

        state = self.get_state(env)
        
        reward = self.score

        # 4. Update environment
        self._update_env(env)

        # 5. Check exit condition
        check = env.goal if env.goal in self.pos else None
        if check:
            self.score[0] += self.p_reward*2
            self.score[1] += self.p_reward*2
            reward[0] = self.score[0]
            reward[1] = self.score[1]
            game_over = True
            return state, reward, game_over, self.score

        # 6. return game over and score
        return state, reward, game_over, self.score

    def get_state(self, env):
        state = np.empty((self.nr,), dtype=np.int8)
        state[0] = self.pos[0].x*env.grid.shape[1] + self.pos[1].y
        state[1] = self.pos[1].x*env.grid.shape[1] + self.pos[0].y
        
        s = [(state[0]*env.grid.shape[0]*env.grid.shape[1] + state[1]),
            (state[1]*env.grid.shape[0]*env.grid.shape[1] + state[0])]
        
        return s
    
    def _is_collision(self, env, i, pt=None):
        if pt is None:
            pt = self.pos[i]
        # hits boundary
        obstacles = np.argwhere(env.grid == 1)
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            return True
        elif pt.y < 0 or pt.y > env.grid.shape[0]-1 or pt.x < 0 or pt.x > env.grid.shape[1]-1:
            self.score[i] -= self.p_reward*2
            return True
        
        return False

    def _is_explored(self, env, pt=None):
        if pt is None:
            pt = self.pos
        # hits boundary
        explored = np.argwhere(env.grid == States.EXP.value)
        if any(np.equal(explored,np.array([self.pos.y,self.pos.x])).all(1)):
            return True
        
        return False
        
    def _update_env(self, env):
        for i in range(0, self.nr):
            if self.frame_iteration == 0:
                # Update robot position(s) on grid
                env.grid[self.pos[i].y,self.pos[i].x] = States.ROBOT.value
            else:
                # Update robot position(s) on grid
                env.grid[self.prev_pos[i].y,self.prev_pos[i].x] = States.EXP.value
                env.grid[self.pos[i].y,self.pos[i].x] = States.ROBOT.value
            

    def _move(self, env, action):
        for i in range(0, self.nr):
            x = self.pos[i].x
            y = self.pos[i].y
            if action[i] == (Direction.RIGHT).value:
                x += 1
            elif action[i] == (Direction.LEFT).value:
                x -= 1
            elif action[i] == (Direction.DOWN).value:
                y += 1
            elif action[i] == (Direction.UP).value:
                y -= 1

            if self._is_collision(env, i, Point(x,y)):
                self.pos[i] = self.pos[i]
                self.prev_pos[i] = self.pos[i]
            else:
                self.prev_pos[i] = self.pos[i]
                self.pos[i] = Point(x,y)

def moving_avarage_smoothing(X,k):
	S = np.zeros(X.shape[0])
	for t in range(X.shape[0]):
		if t < k:
			S[t] = np.mean(X[:t+1])
		else:
			S[t] = np.sum(X[t-k:t])/k
	return S

# Calculates average rewards and steps
def calc_avg(rewards, num_epochs, num_exp, nr):
    avg_rewards = np.sum(np.array(rewards), axis=0)

    avg_rewards = np.divide(avg_rewards, num_epochs)

    mov_avg_rewards = np.empty(avg_rewards.shape)

    for i in range(0, num_exp):
        for j in range(0, nr):
            mov_avg_rewards[i,j] = moving_avarage_smoothing(avg_rewards[i,j], 50)

    return mov_avg_rewards.tolist()

def extract_values(policy_extraction, correct_path, policy, env):
    f = open(os.path.join(correct_path,"saved_data.txt"), "r")
    lines = f.readlines()

    for line in lines:
        cur_line = []
        for char in line:
            if char.isdigit():
                cur_line.append(char)

    if policy_extraction:
        file_name0 = "policy" + str(policy) + "," + str(0) + ".txt"
        file_name1 = "policy" + str(policy) + "," + str(1) + ".txt"
        return np.loadtxt(os.path.join(correct_path, file_name0)), np.loadtxt(os.path.join(correct_path, file_name1)), cur_line
    else:
        return None, cur_line

                    