import numpy as np
import random
import matplotlib.pyplot as plt
import math
import winsound

from cen_m_environment import Environment, HEIGHT, WIDTH, States, Direction, Point
from cen_save_results import print_results

from datetime import datetime
import os
import time
import shutil

class QLearning:
    def __init__(self,env):
        self.score = 0
        self.frame_iteration = 0
        self.nr = 2
        self.starting_pos = env.starting_pos.copy()
        self.pos = env.pos.copy()
        self.prev_pos = env.prev_pos.copy()

        indices = np.argwhere(env.grid == States.UNEXP.value)
        np.random.shuffle(indices)
        self.goal = (Point(indices[0,0], indices[0,1]))

    def run_qlearning(self, mode):
        # Creating The Environment
        env = Environment(self.nr)

        # Creating The Q-Table
        action_space_size = len(Direction)
        state_space_size = (env.grid.shape[1] * env.grid.shape[0])**2

        q_table = np.zeros((state_space_size, action_space_size))

        # Initializing Q-Learning Parameters
        num_episodes = 4000
        max_steps_per_episode = 200
        epochs = 1000

        learning_rate = np.array([0.05, 0.1, 0.2])
        discount_rate = np.array([0.9])

        # pos_reward = env.grid.shape[0]*env.grid.shape[1]
        pos_reward = 2
        n_tests = 100
        interval = 50

        exploration_rate = np.array([0.01], dtype=np.float32)
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
        experiments = len(learning_rate) * len(discount_rate) * len(exploration_rate)
        exp = 0
        epoch_cnt = 0
        
        rewards_per_episode = []
        exp_rewards = []
        avg_rewards = []
        all_rewards = []
        all_grids = []
        action = np.empty((self.nr,), dtype=np.int8)
        q_tables = np.zeros((experiments, state_space_size, action_space_size))

        print("\n# Epochs: ", epochs, "\n# Experiments: ", experiments, "\n# Episodes per experiment: ", num_episodes)
        print("\nHyperparameters:\nLearning rate (α): ", learning_rate, "\nDiscount rate (γ): ", discount_rate, "\nExploration rate (ϵ): ", exploration_rate, "\nExploration decay rate: ", exploration_decay_rate, "\n")

        PATH = os.getcwd()
        PATH = os.path.join(PATH, 'SAR')
        PATH = os.path.join(PATH, 'Results')
        PATH = os.path.join(PATH, 'QLearning')
        PATH = os.path.join(PATH, 'Centralized')
        date_and_time = datetime.now()
        save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
        if not os.path.exists(save_path): os.makedirs(save_path)

        if mode == 1:
            # Training loop
            same_state = []
            training_time = time.time()
            generate = True
            state = self.reset(env, generate)
            generate = False
            for epoch_i in range(0, epochs):
                print("Epoch: ", epoch_cnt)
                exp_rewards = []
                exp = 0
                for lr_i in np.arange(len(learning_rate)):
                    for dr_i in np.arange(len(discount_rate)):
                        exp_exploration_rate = np.copy(exploration_rate)
                        for er_i in np.arange(len(exploration_rate)):
                            print("Experiment: ", exp)
                            # Reinitialize some variables
                            q_table = np.zeros((state_space_size, action_space_size))

                            # Training Loop
                            rewards_per_episode = []

                            # Q-learning algorithm
                            for episode in range(num_episodes):
                                if episode % 100000 == 0:
                                    print("Episode: %s" %(episode))

                                # Initialize new episode params
                                if not episode == 0: state = self.reset(env, generate)
                                done = False
                                rewards_current_episode = [0]*self.nr
                                reward = [0]*self.nr

                                for step in range(max_steps_per_episode):
                                    if state[0] == state[1]:
                                        if state not in same_state:
                                            same_state.append(state)
                                            
                                    # Exploration-exploitation trade-off
                                    exploration_rate_threshold = random.uniform(0, 1)
                                    for i in range(0, self.nr):
                                        if exploration_rate_threshold > exp_exploration_rate[er_i]:
                                            action[i] = np.argmax(q_table[state[i],:])
                                        else:
                                            action_space_actual_size = action_space_size-1
                                            action[i] = random.randint(0, action_space_actual_size)

                                    # Take new action
                                    new_state, reward, done, info = self.step(env, action, pos_reward)

                                    # Update Q-table
                                    q_table[state[0], action[0]] = q_table[state[0], action[0]] * (1 - learning_rate[lr_i]) + \
                                        learning_rate[lr_i] * (reward[0] + discount_rate[dr_i] * np.max(q_table[new_state[0], :]))
                                    q_table[state[1], action[1]] = q_table[state[1], action[1]] * (1 - learning_rate[lr_i]) + \
                                        learning_rate[lr_i] * (reward[0] + discount_rate[dr_i] * np.max(q_table[new_state[1], :]))

                                    # Set new state
                                    state = new_state
                                    
                                    # Add new reward  
                                    # rewards_current_episode += reward
                                    rewards_current_episode[0] += reward[0]
                                    rewards_current_episode[1] += reward[1]

                                    if done:
                                        rewards_per_episode.append(rewards_current_episode)
                                        break

                                    # Exploration rate decay 
                                    exp_exploration_rate[er_i] = min_exploration_rate[er_i] + \
                                        (max_exploration_rate[er_i] - min_exploration_rate[er_i]) * np.exp(-exploration_decay_rate[er_i]*episode)

                                # Add current episode reward to total rewards list
                                if not done:
                                    rewards_per_episode.append(rewards_current_episode)

                            rewards_per_episode = rewards_per_episode[::interval]
                            tmp_exp_rewards = np.array(exp_rewards)
                            new_tmp_exp_rewards = np.array(np.append(tmp_exp_rewards.ravel(),np.array(rewards_per_episode)))
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
            min, sec = divmod(training_time, 60)
            hour = 0
            if min >= 60: hour, min = divmod(min, 60)
            print("Time to train policy: %sh, %sm %ss\nTraining session: %s" %(hour, min, sec, str(save_path)))
            print(same_state)

            results = print_results(env.grid, HEIGHT, WIDTH)
            self.reset(env, generate)

            trajs0 = []
            trajs1 = []	
            for i in range(0, q_tables.shape[0]):	
                print("\nTrajectories of policy %s:" %(i))	
                step_tab_0 = np.zeros((env.grid.shape[0], env.grid.shape[1], env.grid.shape[0], env.grid.shape[1]))
                step_tab_1 = np.zeros((env.grid.shape[0], env.grid.shape[1], env.grid.shape[0], env.grid.shape[1]))


                self.reset(env, generate)	
                for step in range(1, 200):	
                    state = self.get_state(env)	
                    action[0] = np.argmax(q_tables[i, state[0],:])	
                    action[1] = np.argmax(q_tables[i, state[1],:])

                    step_tab_0[self.pos[1].y, self.pos[1].x, self.pos[0].y, self.pos[0].x] = step
                    step_tab_1[self.pos[0].y, self.pos[0].x, self.pos[1].y, self.pos[1].x] = step

                    new_state, reward, done, _ = self.step(env, action, pos_reward)
                    if done:
                        break
                	
                trajs0.append(step_tab_0)
                trajs1.append(step_tab_1)

            results.plot_and_save(q_tables, avg_rewards, epochs,
                                    learning_rate, discount_rate, exploration_rate,
                                    min_exploration_rate, max_exploration_rate, exploration_decay_rate,
                                    save_path, env, hour, min, sec, interval, trajs0, trajs1)

            for i in range(0, q_tables.shape[0]):	
                print("\nTesting policy %s:" % (i))	
                nb_success = [0]*n_tests	
                state_cnter = 0	
                # Display percentage successes	
                for j in range(0, n_tests):	
                    if j % 10 == 0: print("Test number %s out of %s" %(j, n_tests))	
                    state_cnter = 0
                    for state0 in range(0, env.grid.shape[0]*env.grid.shape[1]**self.nr):
                        for state1 in range(0, env.grid.shape[0]*env.grid.shape[1]**self.nr):
                            # initialize new episode params	
                            _ = self.reset(env, True)	
                            	
                            state = [state0, state1]	
                            if not state0 == state1:	
                                state_cnter += 1	
                                for step in range(1, max_steps_per_episode+1):	
                                    # Choose action with highest Q-value for current state (Greedy policy)     	
                                    # Take new action	
                                    action[0] = np.argmax(q_tables[i, state[0],:])	
                                    action[1] = np.argmax(q_tables[i, state[1],:])	
                                    new_state, reward, done, _ = self.step(env, action, pos_reward)	
                                    	
                                    if done:   	
                                        nb_success[j] += 1	
                                        break	
                                    # Set new state	
                                    state = new_state	
                    # Success rate for test j	
                    nb_success[j] = nb_success[j]/state_cnter*100	
                    	
                    	
                # Success rate for all tests	
                success = sum(nb_success)/n_tests

                # Success rate for policy i	
                print ("Success rate of policy %s: %s%%" % (i, success))
            
            winsound.Beep(1000, 500)

            debug_q2 = input("See optimal policy?\nY/N?")
            debug_flag2 = str(debug_q2)

        if mode == 2 or debug_flag2 == 'Y' or debug_flag2 == 'y':
            debug_q3 = input("Policy number?")
            policy = int(debug_q3)
            debug_q4 = input("Test policy?")
            test_policy = int(debug_q4)
            load_path = save_path
            if mode == 2:
                debug_q3 = input("Enter path:")
                folder = str(debug_q3)
                load_path = PATH = os.path.join(PATH, folder)
            q_table_list, env_shape = extract_values(load_path, policy)

            q_table = np.array(q_table_list)
            env.grid = np.zeros((int(env_shape[0]), int(env_shape[1])))

            if test_policy:
                print("\nTesting policy %s:" %(policy))
                nb_success = [0]*n_tests	
                state_cnter = 0	
                # Display percentage successes	
                for j in range(0, n_tests):	
                    if j % 10 == 0: print("Test number %s out of %s" %(j, n_tests))	
                    state_cnter = 0	
                    for state0 in range(0, env.grid.shape[0]*env.grid.shape[1]**self.nr):	
                        for state1 in range(0, env.grid.shape[0]*env.grid.shape[1]**self.nr):	
                            # initialize new episode params	
                            _ = self.reset(env, True)	
                                
                            state = [state0, state1]	
                            if not state0 == state1:	
                                state_cnter += 1	
                                for step in range(1, max_steps_per_episode+1):	
                                    # Choose action with highest Q-value for current state (Greedy policy)     	
                                    # Take new action	
                                    action[0] = np.argmax(q_table[state[0],:])	
                                    action[1] = np.argmax(q_table[state[1],:])	
                                    new_state, reward, done, _ = self.step(env, action, pos_reward)	
                                        
                                    if done:   	
                                        nb_success[j] += 1	
                                        break	
                                    # Set new state	
                                    state = new_state	
                    # Success rate for test j	
                    nb_success[j] = nb_success[j]/state_cnter*100
                        
                # Success rate for all tests	
                success = sum(nb_success)/n_tests

                # Success rate for policy i	
                print ("Success rate of policy %s: %s%%" % (policy, success))

            print("\nTrajectories of policy %s:" %(policy))	
            trajs = []	
            test_tab = np.empty((env.grid.shape[0], env.grid.shape[1]), dtype=str)	
            self.reset(env, generate)	
            for step in range(1, 200):	
                state = self.get_state(env)	
                action[0] = np.argmax(q_table[state[0],:])	
                action[1] = np.argmax(q_table[state[1],:])

                if action[0] == Direction.RIGHT.value:	
                    test_tab[self.pos[0].y, self.pos[0].x] = ">"	
                elif action[0] == Direction.LEFT.value: 	
                    test_tab[self.pos[0].y, self.pos[0].x] = "<"	
                elif action[0] == Direction.UP.value: 	
                    test_tab[self.pos[0].y, self.pos[0].x] = "^"	
                elif action[0] == Direction.DOWN.value: 	
                    test_tab[self.pos[0].y, self.pos[0].x] = "v"

                if action[1] == Direction.RIGHT.value:	
                    test_tab[self.pos[1].y, self.pos[1].x] = ">"	
                elif action[1] == Direction.LEFT.value: 	
                    test_tab[self.pos[1].y, self.pos[1].x] = "<"	
                elif action[1] == Direction.UP.value: 	
                    test_tab[self.pos[1].y, self.pos[1].x] = "^"	
                elif action[1] == Direction.DOWN.value: 	
                    test_tab[self.pos[1].y, self.pos[1].x] = "v"	
                new_state, reward, done, _ = self.step(env, action, pos_reward)	
            	
            trajs.append(test_tab)	
            print(test_tab)


            # Save images
            state = self.reset(env, generate)
            done = False

            for step in range(max_steps_per_episode):
                all_grids.append(env.grid.copy())       
                # Show current state of environment on screen
                # Choose action with highest Q-value for current state (Greedy policy)     
                for i in range(0, self.nr):  
                    # Take new action
                    action[i] = np.argmax(q_table[state[i],:])
                new_state, reward, done, _ = self.step(env, action, pos_reward)
                
                if done:
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
            self.starting_pos = env.starting_pos.copy()	
            self.pos = env.pos.copy()	
            self.prev_pos = env.prev_pos.copy()

            self.goal = env.goal
            env.grid[self.goal.y, self.goal.x] = States.GOAL.value

            self._update_env(env)
        else:
            # Varied starting pos and goal
            # Clear all visited blocks
            env.grid.fill(0)

            np.random.seed()

            # Set goal position  
            indices = np.argwhere(env.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            self.goal = Point(indices[0,0], indices[0,1])
            env.grid[self.goal.y, self.goal.x] = States.GOAL.value

            # Setup agent
            # Set robot(s) start position
            indices = np.argwhere(env.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            self.starting_pos[0] = Point(indices[0,0], indices[0,1])
            self.starting_pos[1] = Point(indices[1,0], indices[1,1])
            
            self.pos = self.starting_pos.copy()
            self.prev_pos = self.starting_pos.copy()

            self._update_env(env)
  
        self.score = 0
        self.frame_iteration = 0

        state = self.get_state(env)

        return state
        
    def step(self, env, action, pos_reward):
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
        # check = self.goal if self.goal in self.pos else None
        # if check:
        if self.pos[0] == self.goal:
            self.score[0] += self.p_reward
            reward[0] = self.score[0]
            game_over = True
            return state, reward, game_over, self.score
        if self.pos[1] == self.goal:
            self.score[1] += self.p_reward
            reward[1] = self.score[1]
            game_over = True
            return state, reward, game_over, self.score

        # 6. return game over and score
        return state, reward, game_over, self.score

    def get_state(self, env):
        state = np.empty((self.nr,), dtype=np.int8)
        state[0] = self.pos[0].y*env.grid.shape[1] + self.pos[0].x
        state[1] = self.pos[1].y*env.grid.shape[1] + self.pos[1].x

        s = [(state[0]*env.grid.shape[0]*env.grid.shape[1] + state[1]),
            (state[1]*env.grid.shape[0]*env.grid.shape[1] + state[0])]
        
        return s
    
    def _is_collision(self, env, pt):
        # hits boundary
        # obstacles = np.argwhere(env.grid == States.OBS.value)
        # if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
        #     return True
        for i in range(0, self.nr):
            if pt.y[i] < 0 or pt.y[i] > env.grid.shape[0]-1 or pt.x[i] < 0 or pt.x[i] > env.grid.shape[1]-1:
                # self.score -= self.p_reward*2
                return True
        if pt.y[0] == pt.y[1] and pt.x[0] == pt.x[1]:
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
        if self.frame_iteration == 0:
            # Update robot position(s) on grid
            for i in range(0, self.nr):
                env.grid[self.pos[i].y,self.pos[i].x] = States.ROBOT.value
        else:
            # Update robot position(s) on grid
            for i in range(0, self.nr): env.grid[self.prev_pos[i].y,self.prev_pos[i].x] = States.EXP.value
            for i in range(0, self.nr): env.grid[self.pos[i].y,self.pos[i].x] = States.ROBOT.value
            

    def _move(self, env, action):
        x = [None]*self.nr
        y = [None]*self.nr
        for i in range(0, self.nr):
            x[i] = self.pos[i].x
            y[i] = self.pos[i].y
            if action[i] == (Direction.RIGHT).value:
                x[i] += 1
            elif action[i] == (Direction.LEFT).value:
                x[i] -= 1
            elif action[i] == (Direction.DOWN).value:
                y[i] += 1
            elif action[i] == (Direction.UP).value:
                y[i] -= 1

        for i in range(0, self.nr):
            if self._is_collision(env, Point(y,x)):
                self.pos[i] = self.pos[i]
                self.prev_pos[i] = self.pos[i]
            else:
                self.prev_pos[i] = self.pos[i]
                self.pos[i] = Point(y[i],x[i])

def moving_avarage_smoothing(X,k):
	S = np.zeros(X.shape[0])
	for t in range(X.shape[0]):
		if t < k:
			S[t] = np.mean(X[:t+1])
		else:
			S[t] = np.sum(X[t-k:t])/k
	return S

# Calculates average rewards and steps
def calc_avg(rewards, num_epochs, num_sims, nr):
    avg_rewards = np.sum(np.array(rewards), axis=0)

    avg_rewards = np.divide(avg_rewards, num_epochs)

    mov_avg_rewards = np.empty(avg_rewards.shape)

    for i in range(0, num_sims):
        for j in range(0, nr):
            mov_avg_rewards[i,j] = moving_avarage_smoothing(avg_rewards[i,j], 50)

    # return mov_avg_rewards.tolist()
    return avg_rewards.tolist()

def extract_values(correct_path, policy):
    f = open(os.path.join(correct_path,"env_shape.txt"), "r")
    lines = f.readlines()

    for line in lines:
        cur_line = []
        for char in line:
            if char.isdigit():
                cur_line.append(char)

    file_name = "policy" + str(policy) + ".txt"
    return np.loadtxt(os.path.join(correct_path, file_name)), cur_line

