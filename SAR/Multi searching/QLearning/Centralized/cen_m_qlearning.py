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

        self.the_fucking_dumb_ass_state = None

    def run_qlearning(self, mode):
        # Creating The Environment
        env = Environment(self.nr)

        # Creating The Q-Table
        action_space_size = len(Direction)
        state_space_size = (env.grid.shape[1] * env.grid.shape[0])**2

        q_table = np.zeros((state_space_size, action_space_size))

        # Initializing Q-Learning Parameters
        num_episodes = 400000
        max_steps_per_episode = 200
        epochs = 1

        learning_rate = np.array([0.01])
        discount_rate = np.array([0.1, 0.5, 0.9])

        # pos_reward = env.grid.shape[0]*env.grid.shape[1]
        pos_reward = 2
        n_tests = 50
        interval = 5000

        exploration_rate = np.array([0.05, 0.1, 0.3], dtype=np.float32)
        max_exploration_rate = np.array([0., 0.1, 0.3], dtype=np.float32)
        min_exploration_rate = np.array([0.05, 0.1, 0.3], dtype=np.float32)
        exploration_decay_rate = np.array([0.05, 0.1, 0.3], dtype=np.float32)

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

        if mode != 1:
            PATH = os.getcwd()
            PATH = os.path.join(PATH, 'SAR')
            PATH = os.path.join(PATH, 'Results')
            PATH = os.path.join(PATH, 'QLearning')
            PATH = os.path.join(PATH, 'Centralized')
            load_path = os.path.join(PATH, 'Saved_data')
            if not os.path.exists(load_path): os.makedirs(load_path)
            date_and_time = datetime.now()
            save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
            if not os.path.exists(save_path): os.makedirs(save_path)
        if mode == 3:
            _, env.grid.shape = extract_values(load_path, None)
        if mode != 2:
            PATH = os.getcwd()
            PATH = os.path.join(PATH, 'SAR')
            PATH = os.path.join(PATH, 'Results')
            PATH = os.path.join(PATH, 'QLearning')
            PATH = os.path.join(PATH, 'Centralized')
            load_path = os.path.join(PATH, 'Saved_data')
            if not os.path.exists(load_path): os.makedirs(load_path)
            date_and_time = datetime.now()
            save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
            if not os.path.exists(save_path): os.makedirs(save_path)

        if mode == 1 or mode == 3:
            self.dumbass_row_bool = False
            self.dumbass = False
            state_checker = []
            # Training loop
            training_time = time.time()
            generate = False
            state = self.reset(env, generate)
            for epoch_i in range(0, epochs):
                print("Epoch: ", epoch_cnt)
                exp_rewards = []
                seq_steps = []
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
                            state_cnt = np.zeros((144,))
                            grid_cnt0 = np.copy(env.grid)
                            grid_cnt0.fill(0)
                            grid_cnt1 = np.copy(env.grid)
                            grid_cnt1.fill(0)
                            the_x1 = None
                            the_y1 = None
                            the_x2 = None
                            the_y2 = None

                            # Q-learning algorithm
                            for episode in range(num_episodes):
                                if episode % 10000 == 0:
                                    print("Episode: %s\nϵ=%s" %(episode, exp_exploration_rate[er_i]))

                                # Initialize new episode params
                                if not episode == 0: state = self.reset(env, generate)
                                done = False
                                rewards_current_episode = 0
                                reward = 0

                                for step in range(max_steps_per_episode):
                                    state_cnt[state[0]] += 1
                                    state_cnt[state[1]] += 1
                                    grid_cnt0[self.pos[0].y, self.pos[0].x] += 1
                                    grid_cnt1[self.pos[1].y, self.pos[1].x] += 1
                                    if self.pos[0] == self.pos[1]:
                                        if state in state_checker:
                                            pass
                                        else:
                                            state_checker.append(state)

                                    if self.dumbass:
                                        self.dumbass = False
                                        for y1 in range(env.grid.shape[0]):
                                            for x1 in range(env.grid.shape[1]):
                                                for y2 in range(env.grid.shape[0]):
                                                    for x2 in range(env.grid.shape[1]):
                                                        s = np.empty((self.nr,), dtype=np.int8)
                                                        s[0] = y1*env.grid.shape[1] + x1
                                                        s[1] = y2*env.grid.shape[1] + x2

                                                        s = [(s[0]*env.grid.shape[0]*env.grid.shape[1] + s[1]),
                                                            (s[1]*env.grid.shape[0]*env.grid.shape[1] + s[0])]

                                                        if s[0] == self.the_fucking_dumb_ass_state:
                                                            the_x1 = x1
                                                            the_y1 = y1
                                                        elif s[1] == self.the_fucking_dumb_ass_state:
                                                            the_x2 = x2
                                                            the_y2 = y2
                                    
                                    if self.pos[0] == Point(the_y1, the_x1) and self.pos[1] == Point(the_y2, the_x2):
                                        print("FUCK")

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

                                    if episode > 20000 and not self.dumbass_row_bool:
                                        for a in range(q_table.shape[0]):
                                            if (q_table[a] == [0,0,0,0]).all():
                                                self.the_fucking_dumb_ass_state = a
                                                self. dumbass_row_bool = True
                                                self. dumbass = True
                                                print(a)

                                    # Update Q-table
                                    q_table[state[0], action[0]] = q_table[state[0], action[0]] * (1 - learning_rate[lr_i]) + \
                                        learning_rate[lr_i] * (reward + discount_rate[dr_i] * np.max(q_table[new_state[0], :]))
                                    q_table[state[1], action[1]] = q_table[state[1], action[1]] * (1 - learning_rate[lr_i]) + \
                                        learning_rate[lr_i] * (reward + discount_rate[dr_i] * np.max(q_table[new_state[1], :]))

                                    # Set new state
                                    state = new_state
                                    
                                    # Add new reward  
                                    rewards_current_episode += reward

                                    if done:
                                        rewards_per_episode.append(rewards_current_episode)
                                        break

                                    # Exploration rate decay 
                                    exp_exploration_rate[er_i] = min_exploration_rate[er_i] + \
                                        (max_exploration_rate[er_i] - min_exploration_rate[er_i]) * np.exp(-exploration_decay_rate[er_i]*episode)

                                # Add current episode reward to total rewards list
                                if mode:
                                    if not done:
                                        rewards_per_episode.append(rewards_current_episode)

                            if mode:
                                rewards_per_episode = rewards_per_episode[::interval]
                                tmp_exp_rewards = np.array(exp_rewards)
                                new_tmp_exp_rewards = np.array(np.append(tmp_exp_rewards.ravel(),np.array(rewards_per_episode)))
                                if tmp_exp_rewards.shape[0] == 0:
                                    new_exp_rewards = new_tmp_exp_rewards.reshape(1,len(rewards_per_episode))
                                else:
                                    new_exp_rewards = new_tmp_exp_rewards.reshape(new_exp_rewards.shape[0]+1, new_exp_rewards.shape[1])
                                exp_rewards = new_exp_rewards.tolist()
                                
                                q_tables[exp] = q_table

                            exp += 1
                tmp_rewards = np.array(all_rewards)
                new_tmp_rewards = np.array(np.append(tmp_rewards.ravel(),np.array(exp_rewards).ravel()))
                new_rewards = new_tmp_rewards.reshape(epoch_cnt+1,exp,int(num_episodes/interval))
                all_rewards = new_rewards.tolist()

                epoch_cnt += 1
            
            avg_rewards = calc_avg(new_rewards, epochs, experiments)
            training_time = time.time() - training_time
            print("Time to train policy: %sm %ss" %(divmod(training_time, 60)))

            results = print_results(env.grid, HEIGHT, WIDTH)
            self.reset(env, generate)

            print(state_cnt)
            print(grid_cnt0)
            print(grid_cnt1)
            print(the_x1, the_y1, the_x2, the_y2, self.the_fucking_dumb_ass_state)
            print(state_checker)

            trajs = []
            for i in range(0, q_tables.shape[0]):
                print("\nTrajectories of policy %s:" %(i))
                test_tab = np.empty((env.grid.shape[0], env.grid.shape[1]), dtype=str)
                self.reset(env, generate)
                for step in range(1, 200):
                    state = self.get_state(env)

                    action[0] = np.argmax(q_tables[i, state[0],:])
                    action[1] = np.argmax(q_tables[i, state[1],:])

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

            results.plot_and_save(q_tables, avg_rewards, learning_rate, discount_rate, exploration_rate, min_exploration_rate, max_exploration_rate, exploration_decay_rate, save_path, env, training_time, trajs, interval)

            for i in range(0, q_tables.shape[0]):
                print("\nTesting policy %s:" % (i))
                nb_success = [0]*n_tests
                zeros = [0]*q_tables.shape[0]
                # Display percentage successes
                for j in range(0, n_tests):
                    if j % 10 == 0: print("Testing...", j)
                    zeros[i] = 0
                    for state0 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                        for state1 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                            # initialize new episode params
                            _ = self.reset(env, True)

                            state = [(state0*env.grid.shape[0]*env.grid.shape[1] + state1),
                                (state1*env.grid.shape[0]*env.grid.shape[1] + state0)]
                            tmp_zeros0 = [a for a in q_tables[i, state[0]] if a == 0]
                            zeros[i] += len(tmp_zeros0)
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

                # Let's check our success rate!
                success = sum(nb_success)/n_tests

                # Let's check our success rate!
                print ("Success rate of policy %s: %s / %s * 100 = %s %%" % (i, success, state_space_size, success/state_space_size*100))
                print("Policy %s agent: %s\nOut of: %s" %(str(i), str(zeros[i]), str(state_space_size*action_space_size)))
            
            winsound.Beep(1000, 500)

            debug_q2 = input("See optimal policy?\nY/N?")
            debug_flag2 = str(debug_q2)

        if mode == 2 or debug_flag2 == 'Y' or debug_flag2 == 'y':
            debug_q3 = input("Policy number?")
            policy = int(debug_q3)
            load_path = save_path
            if mode == 2:
                debug_q3 = input("Enter path:")
                folder = str(debug_q3)
                load_path = PATH = os.path.join(PATH, folder)
            q_table_list, env_shape = extract_values(load_path, policy)

            q_table = np.array(q_table_list)
            env.grid = np.zeros((int(env_shape[0]), int(env_shape[1])))

            if mode == 2:
                print("\nTesting policy %s:" % (policy))
                nb_success = 0
                # Display percentage successes
                # for episode in range(10000):
                #     if episode % 1000 == 0: print("Episode: ", episode)
                #     # initialize new episode params
                #     state = self.reset(env, generate)
                #     #print(env.grid)
                #     #print(env.grid)
                #     done = False
                #     for step in range(max_steps_per_episode):   
                #         # Show current state of environment on screen
                #         # Choose action with highest Q-value for current state (Greedy policy)     
                #         for i in range(0, self.nr):  
                #             # Take new action
                #             action[i] = np.argmax(q_table[state[i],:])
                #         new_state, reward, done, info = self.step(env, action, pos_reward)
                        
                #         if done:   
                #             nb_success += 1
                #             break

                #         # Set new state
                #         state = new_state

                # # Let's check our success rate!
                # print ("Success rate of policy %s = %s %%" % (policy, nb_success/100))

            print("\nTrajectories of policy %s:" %(policy))
            test_tab = [None] * (env.grid.shape[1]*env.grid.shape[0])
            for s in range(env.grid.shape[0]*env.grid.shape[1]):
                a = np.argmax(q_table[s,:])
                if a == Direction.RIGHT.value:
                    test_tab[s] = ">"
                elif a == Direction.LEFT.value: 
                    test_tab[s] = "<"
                elif a == Direction.UP.value: 
                    test_tab[s] = "^"
                elif a == Direction.DOWN.value: 
                    test_tab[s] = "v"
        
            print(np.reshape(test_tab, (env.grid.shape[1], env.grid.shape[0])).T)

            state = self.reset(env, generate)
            # env.grid[self.pos.y, self.pos.x] = States.UNEXP.value
            # self.prev_pos = self.starting_pos
            # self.pos = self.prev_pos
            # env.grid[self.pos.y, self.pos.x] = States.ROBOT.value
            done = False

            for step in range(max_steps_per_episode):
                if mode: all_grids.append(env.grid.copy())       
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
            self.prev_pos = self.starting_pos.copy()
            self.pos = self.prev_pos.copy()

            self.goal = env.goal
            env.grid[self.goal.y, self.goal.x] = States.GOAL.value

            self._update_env(env)
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
            # Clear all visited blocks
            env.grid.fill(0)

            # Setup agent
            # Set robot(s) start position
            indices = np.argwhere(env.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            self.starting_pos[0] = Point(indices[0,0], indices[0,1])
            self.starting_pos[1] = Point(indices[1,0], indices[1,1])
            
            self.pos = self.starting_pos.copy()

            self._update_env(env)

            # Set goal position        
            env.grid[self.goal.y, self.goal.x] = States.GOAL.value
                
        self.score = 0
        self.frame_iteration = 0

        state = self.get_state(env)

        return state
        
    def step(self, env, action, pos_reward):
        #print(self.pos, self.prev_pos, action)
        #print(self.grid,"\n")
        self.frame_iteration += 1
        self.p_reward = pos_reward
        self.score = 0
        # 2. Do action
        self._move(env, action) # update the robot
            
        # 3. Update score and get state
        self.score -= 0.1
        game_over = False

        state = self.get_state(env)
        
        reward = self.score

        # 4. Update environment
        self._update_env(env)

        # 5. Check exit condition
        check = self.goal if self.goal in self.pos else None
        if check:
            self.score += self.p_reward
            reward = self.score
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
    
    def _is_collision(self, env, i, pt=None):
        if pt is None:
            pt = self.pos[i]
        # hits boundary
        obstacles = np.argwhere(env.grid == States.OBS.value)
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            return True
        elif pt.y < 0 or pt.y > env.grid.shape[0]-1 or pt.x < 0 or pt.x > env.grid.shape[1]-1:
            self.score -= self.p_reward*2
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

            if self._is_collision(env, i, Point(y,x)):
                self.pos[i] = self.pos[i]
                self.prev_pos[i] = self.pos[i]
            else:
                self.prev_pos[i] = self.pos[i]
                self.pos[i] = Point(y,x)

def moving_avarage_smoothing(X,k):
	S = np.zeros(X.shape[0])
	for t in range(X.shape[0]):
		if t < k:
			S[t] = np.mean(X[:t+1])
		else:
			S[t] = np.sum(X[t-k:t])/k
	return S

# Calculates average rewards and steps
def calc_avg(rewards, num_epochs, num_sims):
    avg_rewards = np.sum(np.array(rewards), axis=0)

    avg_rewards = np.divide(avg_rewards, num_epochs)

    mov_avg_rewards = np.empty(avg_rewards.shape)

    for i in range(0, num_sims):
        mov_avg_rewards[i] = moving_avarage_smoothing(avg_rewards[i], 50)

    return mov_avg_rewards.tolist()

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

