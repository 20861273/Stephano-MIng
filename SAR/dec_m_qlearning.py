from pickletools import uint8
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import winsound

from m_environment import Environment, HEIGHT, WIDTH, States, Direction, Point
from save_results import print_results

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
        state_space_size = (env.grid.shape[1] * env.grid.shape[0])
        #print(state_space_size, action_space_size)

        q_table = np.zeros((self.nr, state_space_size, action_space_size))

        # Initializing Q-Learning Parameters
        num_episodes = 200000
        max_steps_per_episode = 200
        num_epochs = 1

        learning_rate = np.array([0.00075])
        discount_rate = np.array([0.002])

        # pos_reward = env.grid.shape[0]*env.grid.shape[1]
        pos_reward = 1
        n_tests = 50

        exploration_rate = np.array([0.03], dtype=np.float32)
        max_exploration_rate = np.array([0.03], dtype=np.float32)
        min_exploration_rate = np.array([0.03], dtype=np.float32)
        exploration_decay_rate = np.array([0.03], dtype=np.float32)

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
        action = np.empty((self.nr,), dtype=np.int8)
        q_tables = np.zeros((num_sims, self.nr, state_space_size, action_space_size))

        print("\n# Epochs: ", num_epochs, "\n# Training sessions per epoch: ", num_sims, "\n# Episodes per training session: ", num_episodes)
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
            load_path = os.path.join(PATH, 'Saved_data')
            if not os.path.exists(load_path): os.makedirs(load_path)
            date_and_time = datetime.now()
            save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
            if not os.path.exists(save_path): os.makedirs(save_path)

        if mode == 1 or mode == 3:
            # Training loop
            training_time = time.time()
            generate = False
            for seq_i in range(0, num_epochs):
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
                            q_table = np.zeros((self.nr, state_space_size, action_space_size))

                            # Training Loop
                            episode_len = []
                            steps_per_episode = []
                            rewards_per_episode = []                

                            # Q-learning algorithm
                            for episode in range(num_episodes):
                                if episode % 10000 == 0: print("Episode: ", episode)
                                
                                # Initialize new episode params
                                state = self.reset(env, generate)
                                done = False
                                rewards_current_episode = 0
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
            
            avg_rewards, avg_steps = calc_avg(new_rewards, new_steps, num_epochs, num_sims)
            training_time = time.time() - training_time
            print("Time to train policy: %sm %ss" %(divmod(training_time, 60)))

            results = print_results(env.grid, HEIGHT, WIDTH)
            self.reset(env, generate)

            # trajs = []
            # for i in range(0, q_tables.shape[0]):
            #     print("\nTrajectories of policy %s:" %(i))
            #     test_tab = [None] * (env.grid.shape[1]*env.grid.shape[0])
            #     for s in range(env.grid.shape[0]*env.grid.shape[1]):
            #         a = np.argmax(q_tables[i,s,:])
            #         if a == Direction.RIGHT.value:
            #             test_tab[s] = ">"
            #         elif a == Direction.LEFT.value: 
            #             test_tab[s] = "<"
            #         elif a == Direction.UP.value: 
            #             test_tab[s] = "^"
            #         elif a == Direction.DOWN.value: 
            #             test_tab[s] = "v"
            
            #     trajs.append(np.reshape(test_tab, (env.grid.shape[1], env.grid.shape[0])).T)
            #     print(np.reshape(test_tab, (env.grid.shape[1], env.grid.shape[0])).T)

            results.plot(q_tables, avg_rewards, avg_steps, learning_rate, discount_rate, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, load_path, env, training_time)

            for i in range(0, q_tables.shape[0]):
                print("\nTesting policy %s:" % (i))
                nb_success = [0]*n_tests
                # Display percentage successes
                for j in range(0, n_tests):
                    if j % 10 == 0: print("Testing...", j)
                    for state0 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                        for state1 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                            # if episode % 1000 == 0: print("Episode: ", episode)
                            # initialize new episode params
                            tmp_state = self.reset(env, True)
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

            winsound.Beep(1000, 500)

            debug_q2 = input("See optimal policy?\nY/N?")
            debug_flag2 = str(debug_q2)

        if mode == 2 or debug_flag2 == 'Y' or debug_flag2 == 'y':
            policy_extraction = True
            debug_q3 = input("Policy number?")
            policy = int(debug_q3)
            if mode == 2: correct_path = load_path
            else: correct_path = save_path
            q_table_list, env_shape = extract_values(policy_extraction, load_path, policy, env)

            q_table = np.array(q_table_list)
            env.grid = np.zeros((int(env_shape[0]), int(env_shape[1])))

            # if mode == 2:
            #     print("\nTesting policy %s:" % (policy))
            #     nb_success = 0
            #     indices = np.argwhere(env.grid == States.UNEXP.value)
            #     np.random.shuffle(indices)
            #     env.goal = (Point(indices[0,1], indices[0,0]))
            #     # Display percentage successes
            #     for episode in range(1000000):
            #         if episode % 10000 == 0: print("Episode: ", episode)
            #         # initialize new episode params
            #         state = self.reset(env, generate)
            #         #print(env.grid)
            #         #print(env.grid)
            #         done = False
            #         for step in range(max_steps_per_episode):   
            #             # Show current state of environment on screen
            #             # Choose action with highest Q-value for current state (Greedy policy)     
            #             for i in range(0, self.nr):  
            #                 # Take new action
            #                 action[i] = np.argmax(q_table[state[i],:])
            #             new_state, reward, done, info = self.step(env, action, pos_reward)
                        
            #             if done:   
            #                 nb_success += 1
            #                 break

            #             # Set new state
            #             state = new_state

            #     # # Let's check our success rate!
            #     print ("Success rate of policy %s: %s/100000*100 = %s %%" % (policy, nb_success, nb_success/1000000*100))
            nb_success = [0]*n_tests
            # Display percentage successes
            for i in range(0, n_tests):
                if i % 10 == 0: print("Testing...", i)
                for state0 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                    for state1 in range(0, env.grid.shape[0]*env.grid.shape[1]):
                        # if episode % 1000 == 0: print("Episode: ", episode)
                        # initialize new episode params
                        tmp_state = self.reset(env, True)
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
                                nb_success[i] += 1
                                break

                            # Set new state
                            state = new_state

            # Let's check our success rate!
            success = sum(nb_success)/n_tests
            print ("Success rate of policy %s = %s %%" % (policy, success/state_space_size*100))
        

            # print("\nTrajectories of policy %s:" %(policy))
            # test_tab = [None] * (env.grid.shape[1]*env.grid.shape[0])
            # for s in range(state_space_size):
            #     a = np.argmax(q_table[s,:])
            #     if a == Direction.RIGHT.value:
            #         test_tab[s] = ">"
            #     elif a == Direction.LEFT.value: 
            #         test_tab[s] = "<"
            #     elif a == Direction.UP.value: 
            #         test_tab[s] = "^"
            #     elif a == Direction.DOWN.value: 
            #         test_tab[s] = "v"
        
            # print(np.reshape(test_tab, (env.grid.shape[1], env.grid.shape[0])).T)

            state = self.reset(env, generate)
            # env.grid[self.pos.y, self.pos.x] = States.UNEXP.value
            # self.prev_pos = self.starting_pos
            # self.pos = self.prev_pos
            # env.grid[self.pos.y, self.pos.x] = States.ROBOT.value
            done = False

            for step in range(1, max_steps_per_episode+1):
                if mode: all_grids.append(env.grid.copy())       
                # Show current state of environment on screen
                # Choose action with highest Q-value for current state (Greedy policy)     
                for i in range(0, self.nr):  
                    # Take new action
                    action[i] = np.argmax(q_table[i, state[i],:])
                # print(env.grid)
                new_state, reward, done, info = self.step(env, action, pos_reward, step)
                # print(self.pos, env.goal)
                
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

        else:   
            if len(os.listdir(save_path)):
                try:
                    shutil.rmtree(save_path)
                except OSError as e:
                    print("Tried to delete folder that doesn't exist.")


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
            self.starting_pos = [0]*self.nr
            indices = np.argwhere(env.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            i = 0
            self.starting_pos[0] = Point(indices[i,1], indices[i,0])
            self.starting_pos[1] = self.starting_pos[0]

            while not self.starting_pos[0].y < env.grid.shape[0]/2:
                self.starting_pos[0] = Point(indices[i,1], indices[i,0])
                i += 1
            while not self.starting_pos[1].y >= env.grid.shape[0]/2:
                self.starting_pos[1] = Point(indices[i,1], indices[i,0])
                i += 1
            
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
        self.score -= 0.1
        game_over = False

        state = self.get_state(env)
        
        reward = self.score

        # 4. Update environment
        self._update_env(env)

        # 5. Check exit condition
        if self.pos[0] == self.goal:
            self.score[0] += self.p_reward*2
            reward[0] = self.score[0]
            game_over = True
            return state, reward, game_over, self.score
        if self.pos[1] == self.goal:
            self.score[1] += self.p_reward*2
            reward[1] = self.score[1]
            game_over = True
            return state, reward, game_over, self.score

        # 6. return game over and score
        return state, reward, game_over, self.score

    def get_state(self, env):
        state = np.empty((self.nr,), dtype=np.int8)
        for i in range(0, self.nr):
            state[i] = (self.pos[i].x*env.grid.shape[0] + self.pos[i].y)
        
        return state

    
    def _is_collision(self, env, i, pt=None):
        if pt is None:
            pt = self.pos[i]
        # hits boundary
        obstacles = np.argwhere(env.grid == 1)
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
def calc_avg(rewards, steps, num_epochs, num_sims):
    avg_rewards = np.sum(np.array(rewards), axis=0)
    avg_steps = np.sum(np.array(steps), axis=0)

    avg_rewards = np.divide(avg_rewards, num_epochs)
    avg_steps = np.divide(avg_steps, num_epochs)

    mov_avg_rewards = np.empty(avg_rewards.shape)
    mov_avg_steps = np.empty(avg_steps.shape)

    for i in range(0, num_sims):
        mov_avg_rewards[i] = moving_avarage_smoothing(avg_rewards[i], 1000)
        # mov_avg_steps[i] = moving_avarage_smoothing(avg_steps[i], 1)

    # return mov_avg_rewards.tolist(), mov_avg_steps.tolist()
    return mov_avg_rewards.tolist(), avg_steps.tolist()

def extract_values(policy_extraction, correct_path, policy, env):
    f = open(os.path.join(correct_path,"saved_data.txt"), "r")
    lines = f.readlines()

    for line in lines:
        cur_line = []
        for char in line:
            if char.isdigit():
                cur_line.append(char)

    if policy_extraction:
        file_name = "policy" + str(policy) + ".txt"
        return np.loadtxt(os.path.join(correct_path, file_name)), cur_line
    else:
        return None, cur_line

                    