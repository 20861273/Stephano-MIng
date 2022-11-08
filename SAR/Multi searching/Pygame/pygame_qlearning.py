from collections import namedtuple
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import winsound

from pygame_environment import Block_state
from save_results import print_results

from datetime import datetime
import os
import time
import shutil

from enum import Enum
import pygame

class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class Agent:
    def __init__(self, environment, input_data):
        self.nr = input_data["nr"]

        self.point = namedtuple('Point', 'x, y')

        self.current_position = [0]*self.nr
        self.previous_position = [0]*self.nr
        self.starting_position = [0]*self.nr

        self.current_position = self.starting_position

        indices = np.argwhere(environment.grid == Block_state.UNEXP.value)
        np.random.shuffle(indices)
        self.goal = (self.point(indices[0,1], indices[0,0]))
        
        self.score = 0
        self.positive_reward = environment.grid.shape[0]*environment.grid.shape[1]

    def reset(self, environment):
        # Varied starting position and constant goal
        # Clear all visited blocks
        visited = np.argwhere(environment.grid == Block_state.EXP.value)
        for i in visited:
            environment.grid[i[0], i[1]] = Block_state.UNEXP.value

        # Setup agent
        # Clear all robot blocks
        robot = np.argwhere(environment.grid == Block_state.ROBOT.value)
        for i in robot:
            environment.grid[i[0], i[1]] = Block_state.UNEXP.value
            
        # Set robot(s) start position
        self.starting_position = [0]*self.nr
        indices = np.argwhere(environment.grid == Block_state.UNEXP.value)
        np.random.shuffle(indices)
        for i in range(0, self.nr):
            self.starting_position[i] = (self.point(indices[i,1], indices[i,0]))
            distance_to = 0
            next_i = i
            if not i == 0:
                while distance_to < environment.grid.shape[0]*0.7:
                    self.starting_position[i] = self.point(indices[next_i,1], indices[next_i,0])
                    distance_to = math.sqrt((self.starting_position[i].x - self.starting_position[i-1].x)**2 +
                                    (self.starting_position[i].y - self.starting_position[i-1].y)**2)
                    next_i += 1

            environment.grid[self.starting_position[i].y, self.starting_position[i].x] = Block_state.ROBOT.value
        
        self.current_position = self.starting_position

        # Set goal position        
        environment.grid[self.goal.y, self.goal.x] = Block_state.GOAL.value

        environment.reset_environment()
                
        self.score = 0
        self.frame_iteration = 0

        state = self.get_state(environment)

        return state

    def step(self, environment, action):
        self.score = 0
        # 2. Do action
        self._move(environment, action) # update the robot
            
        # 3. Update score and get state
        self.score -= 0.1
        game_over = False

        state = self.get_state(environment)
        
        reward = self.score

        # 4. Update environment
        environment.update_environment(self.nr, self.current_position, self.previous_position)

        self.frame_iteration += 1

        # 5. Check exit condition
        if self.current_position == self.goal:
            self.score += self.positive_reward
            reward = self.score
            game_over = True
            return state, reward, game_over, self.score

        # 6. return game over and score
        return state, reward, game_over, self.score

    def get_state(self, environment):
        state = np.empty((self.nr,), dtype=np.int8)
        for i in range(0, self.nr):
            state[i] = self.current_position[i].x*environment.grid.shape[0] + self.current_position[i].y
        return state
    
    def _is_collision(self, environment, pt=None):
        if pt is None:
            pt = self.current_position
        
        # Hits obstacle or boundary
        obstacles = np.argwhere(environment.grid == Block_state.OBS.value)
        if any(np.equal(obstacles,np.array([pt.y,pt.x])).all(1)):
            return True
        elif pt.y < 0 or pt.y > environment.grid.shape[0]-1 or pt.x < 0 or pt.x > environment.grid.shape[1]-1:
            self.score -= self.positive_reward*2
            return True
        
        return False

    def _is_explored(self, environment, pt=None):
        if pt is None:
            pt = self.current_position
        
        # Checks all blocks in environment if it has been explored
        explored = np.argwhere(environment.grid == Block_state.EXP.value)
        if any(np.equal(explored,np.array([self.current_position.y,self.current_position.x])).all(1)):
            return True
        
        return False            

    def _move(self, environment, action):
        for i in range(0, self.nr):
            x = self.current_position[i].x
            y = self.current_position[i].y

            if action == Direction.LEFT.value:
                x += 1
            elif action == Direction.RIGHT.value:
                x -= 1
            elif action == Direction.UP.value:
                y += 1
            elif action == Direction.DOWN.value:
                y -= 1

            if self._is_collision(environment, self.point(x,y)):
                self.current_position[i] = self.current_position[i]
                self.previous_position[i] = self.current_position[i]
            else:
                self.previous_position[i] = self.current_position[i]
                self.current_position[i] = self.point(x,y)

class QLearning:
    def __init__(self, environment, input_data):
        self.agent = Agent(environment, input_data)

        # Creating The Q-Table
        self.action_space_size = len(Direction)
        self.state_space_size = environment.grid.shape[1] * environment.grid.shape[0]

        self.q_table = np.zeros((self.state_space_size, self.action_space_size))

        # Initializing Q-Learning Parameters
        self.num_episodes = input_data["hyperparameters"]["episodes"]
        self.max_steps_per_episode = input_data["hyperparameters"]["max_steps"]
        self.num_epochs = input_data["hyperparameters"]["epochs"]

        self.learning_rate = np.array(input_data["hyperparameters"]["learning_rate"])
        self.discount_rate = np.array(input_data["hyperparameters"]["discount_rate"])
        self.pos_reward = environment.grid.shape[0]*environment.grid.shape[1]

        self.exploration_rate = np.array(input_data["hyperparameters"]["exploration_rate"], dtype=np.float32)
        self.max_exploration_rate = np.array(input_data["hyperparameters"]["max_exploration_rate"], dtype=np.float32)
        self.min_exploration_rate = np.array(input_data["hyperparameters"]["min_exploration_rate"], dtype=np.float32)
        self.exploration_decay_rate = np.array(input_data["hyperparameters"]["exploration_decay_rate"], dtype=np.float32)

    def run_qlearning(self, mode, environment):
        generate = True
        policy_extraction = True
        if mode == 2:
            generate = False
        elif mode == 3:
            generate = False
            policy_extraction = False

        # Result parameters
        num_sims = len(self.learning_rate) * len(self.discount_rate) * len(self.exploration_rate)
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
        q_tables = np.zeros((num_sims, self.state_space_size, self.action_space_size))

        print("\n# Epochs: ", self.num_epochs, "\n# Training sessions per epoch: ", num_sims, "\n# Episodes per training session: ", self.num_episodes)
        print("\nHyperparameters:\nLearning rate (α): ", self.learning_rate, "\nDiscount rate (γ): ", self.discount_rate, "\nExploration rate (ϵ): ", self.exploration_rate, "\nExploration decay rate: ", self.exploration_decay_rate, "\n")

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
            _, environment.grid.shape = extract_values(policy_extraction, load_path, None, environment)
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
        debug_wait = False
        if mode == 1 or mode == 3:
            # Training loop
            training_time = time.time()
            generate = False
            state = self.agent.reset(environment)
            done_training = False
            while not done_training:
                clock = pygame.time.Clock()
                clock.tick(60)
                for event in pygame.event.get():  # User did something
                    if event.type == pygame.QUIT:  # If user clicked close
                        done_training = True
                    for seq_i in range(0, self.num_epochs):
                        print("Training session: ", seq)
                        seq_rewards = []
                        seq_steps = []
                        sim = 0
                        for lr_i in np.arange(len(self.learning_rate)):
                            for dr_i in np.arange(len(self.discount_rate)):
                                for er_i in np.arange(len(self.exploration_rate)):
                                    print("Simulation: ", sim)
                                    # Reinitialize some variables
                                    ep_exploration_rate = np.copy(self.exploration_rate)
                                    q_table = np.zeros((self.state_space_size, self.action_space_size))

                                    # Training Loop
                                    episode_len = []
                                    steps_per_episode = []
                                    rewards_per_episode = []                

                                    # Q-learning algorithm
                                    for episode in range(self.num_episodes):
                                        if episode % 10000 == 0: print("Episode: ", episode)
                                        
                                        # Initialize new episode params
                                        state = self.agent.reset(environment)
                                        done = False
                                        rewards_current_episode = 0
                                        reward = 0

                                        for step in range(self.max_steps_per_episode):
                                            # Exploration-exploitation trade-off
                                            exploration_rate_threshold = random.uniform(0, 1)
                                            if exploration_rate_threshold > ep_exploration_rate[er_i]:
                                                action = np.argmax(q_table[state,:])
                                            else:
                                                action_space_actual_size = self.action_space_size-1
                                                action = random.randint(0, action_space_actual_size)

                                            # Take new action
                                            new_state, reward, done, info = self.agent.step(environment, action)

                                            if event.type == pygame.MOUSEBUTTONDOWN or not debug_wait:
                                                debug_wait = False
                                                while not debug_wait:
                                                    if event.type == pygame.MOUSEBUTTONDOWN: debug_wait = True


                                            # Update Q-table
                                            q_table[state, action] = q_table[state, action] * (1 - self.learning_rate[lr_i]) + \
                                                self.learning_rate[lr_i] * (reward + self.discount_rate[dr_i] * np.max(q_table[new_state, :]))

                                            # Set new state
                                            state = new_state
                                            
                                            # Add new reward  
                                            rewards_current_episode += reward

                                            if done == True:
                                                rewards_per_episode.append(rewards_current_episode)
                                                steps_per_episode.append(step+1)
                                                break

                                            # Exploration rate decay (Exponential decay)
                                            ep_exploration_rate[er_i] = self.min_exploration_rate[er_i] + \
                                                (self.max_exploration_rate[er_i] - self.min_exploration_rate[er_i]) * np.exp(-self.exploration_decay_rate[er_i]*episode)

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
                new_rewards = new_tmp_rewards.reshape(seq+1,sim,self.num_episodes)
                all_rewards = new_rewards.tolist()

                tmp_steps = np.array(all_steps)
                new_tmp_steps = np.array(np.append(tmp_steps.ravel(),np.array(seq_steps).ravel()))
                new_steps = new_tmp_steps.reshape(seq+1,sim,self.num_episodes)
                all_steps = new_steps.tolist()

                seq += 1
                done_training = True
            
            avg_rewards, avg_steps = calc_avg(new_rewards, new_steps, self.num_epochs, num_sims)
            training_time = time.time() - training_time
            print("Time to train policy: %sm %ss" %(divmod(training_time, 60)))

            results = print_results(environment.grid, environment.height, environment.width)
            self.agent.reset(environment)
            trajs = []

            for i in range(0, q_tables.shape[0]):
                print("\nTrajectories of policy %s:" %(i))
                test_tab = [None] * (environment.grid.shape[1]*environment.grid.shape[0])
                for s in range(environment.grid.shape[0]*environment.grid.shape[1]):
                    a = np.argmax(q_tables[i,s,:])
                    if a == Direction.RIGHT.value:
                        test_tab[s] = ">"
                    elif a == Direction.LEFT.value: 
                        test_tab[s] = "<"
                    elif a == Direction.UP.value: 
                        test_tab[s] = "^"
                    elif a == Direction.DOWN.value: 
                        test_tab[s] = "v"

                trajs.append(np.reshape(test_tab, (environment.grid.shape[1], environment.grid.shape[0])).T)
                print(np.reshape(test_tab, (environment.grid.shape[1], environment.grid.shape[0])).T)

            results.plot(q_tables, avg_rewards, avg_steps, self.learning_rate, self.discount_rate, self.exploration_rate, load_path, environment, training_time, trajs)

            

            for i in range(0, q_tables.shape[0]):
                print("\nTesting policy %s:" % (i))
                nb_success = 0
                # Display percentage successes
                for episode in range(10000):
                    if episode % 1000 == 0: print("Episode: ", episode)
                    # initialize new episode params
                    state = self.agent.reset(environment)
                    #print(environment.grid)
                    #print(environment.grid)
                    done = False
                    for step in range(self.max_steps_per_episode):   
                        # Show current state of environment on screen
                        # Choose action with highest Q-value for current state (Greedy policy)     
                        # Take new action
                        action = np.argmax(q_tables[i, state,:])
                        new_state, reward, done, info = self.agent.step(environment, action)
                        
                        if done:   
                            nb_success += 1
                            break

                        # Set new state
                        state = new_state

                # Let's check our success rate!
                print ("Success rate of policy %s = %s %%" % (i, nb_success/100))
            
            winsound.Beep(1000, 500)

            debug_q2 = input("See optimal policy?\nY/N?")
            debug_flag2 = str(debug_q2)

        if mode == 2 or debug_flag2 == 'Y' or debug_flag2 == 'y':
            policy_extraction = True
            debug_q3 = input("Policy number?")
            policy = int(debug_q3)
            if mode == 2: correct_path = load_path
            else: correct_path = save_path
            q_table_list, env_shape = extract_values(policy_extraction, load_path, policy, environment)

            q_table = np.array(q_table_list)
            environment.grid = np.empty((int(env_shape[0]), int(env_shape[1])))

            if mode == 2:
                print("\nTesting policy %s:" % (policy))
                nb_success = 0
                # Display percentage successes
                for episode in range(10000):
                    if episode % 1000 == 0: print("Episode: ", episode)
                    # initialize new episode params
                    state = self.agent.reset(environment)
                    #print(environment.grid)
                    #print(environment.grid)
                    done = False
                    for step in range(self.max_steps_per_episode):   
                        # Show current state of environment on screen
                        # Choose action with highest Q-value for current state (Greedy policy)     
                        # Take new action
                        action = np.argmax(q_table[state,:])
                        new_state, reward, done, info = self.agent.step(environment, action)
                        
                        if done:   
                            nb_success += 1
                            break

                        # Set new state
                        state = new_state

                # Let's check our success rate!
                print ("Success rate of policy %s = %s %%" % (policy, nb_success/100))

            print("\nTrajectories of policy %s:" %(policy))
            test_tab = [None] * (environment.grid.shape[1]*environment.grid.shape[0])
            for s in range(environment.grid.shape[0]*environment.grid.shape[1]):
                a = np.argmax(q_table[s,:])
                if a == Direction.RIGHT.value:
                    test_tab[s] = ">"
                elif a == Direction.LEFT.value: 
                    test_tab[s] = "<"
                elif a == Direction.UP.value: 
                    test_tab[s] = "^"
                elif a == Direction.DOWN.value: 
                    test_tab[s] = "v"
        
            print(np.reshape(test_tab, (environment.grid.shape[1], environment.grid.shape[0])).T)

            state = self.agentreset(environment)
            environment.grid[environment.pos.y, environment.pos.x] = Block_state.UNEXP.value
            environment.prev_pos = environment.starting_pos
            environment.pos = environment.prev_pos
            environment.grid[environment.pos.y, environment.pos.x] = Block_state.ROBOT.value
            done = False

            for step in range(self.max_steps_per_episode):
                if mode: all_grids.append(environment.grid.copy())       
                # Show current state of environment on screen
                # Choose action with highest Q-value for current state (Greedy policy)     
                # Take new action
                action = np.argmax(q_table[state,:]) 
                new_state, reward, done, info = self.agent.step(environment, action)
                
                if done:
                    if mode: all_grids.append(environment.grid.copy())
                    break

                # Set new state
                state = new_state

                   
            for i in np.arange(len(all_grids)):
                PR = print_results(all_grids[i], environment.grid.shape[0], environment.grid.shape[1])
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
        mov_avg_rewards[i] = moving_avarage_smoothing(avg_rewards[i], 100)
        mov_avg_steps[i] = moving_avarage_smoothing(avg_steps[i], 100)

    return mov_avg_rewards.tolist(), mov_avg_steps.tolist()
    # return avg_rewards.tolist(), avg_steps.tolist()

def extract_values(policy_extraction, correct_path, policy, environment):
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

                    