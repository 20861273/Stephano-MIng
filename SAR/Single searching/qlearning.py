import numpy as np
import random
import matplotlib.pyplot as plt
import math
import winsound

from environment import Environment, HEIGHT, WIDTH, States, Direction, Point
from save_results import print_results, plot_learning_curve, write_json

from datetime import datetime
import os
import time
import shutil

class QLearning:
    def __init__(self):
        self.score = 0
        self.frame_iteration = 0

    def run_qlearning(self, mode):
        # Number of seekers
        fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))

        # Creating The Environment
        env = Environment()
        QL = QLearning()
        PR = print_results(env.grid,HEIGHT,WIDTH)

        # Creating The Q-Table
        action_space_size = len(Direction)
        state_space_size = env.grid.shape[1] * env.grid.shape[0]
        #print(state_space_size, action_space_size)

        # q_table = np.zeros((WIDTH, HEIGHT, WIDTH, HEIGHT, action_space_size))
        q_table = np.zeros((WIDTH, HEIGHT, action_space_size))

        # debugging
        show_plot = False
        print_interval = 1000

        # Initializing Q-Learning Parameters
        num_episodes = 200000
        max_steps_per_episode = WIDTH*HEIGHT*3
        num_epochs = 3

        learning_rate = np.array([0.0001])
        discount_rate = np.array([0.7,0.8,0.9])

        # learning_rate = np.array([0.00075])
        # discount_rate = np.array([0.002])

        pos_reward = 1

        exploration_rate = np.array([0.1], dtype=np.float32)
        max_exploration_rate = np.array([0.1], dtype=np.float32)
        min_exploration_rate = np.array([0.1], dtype=np.float32)
        exploration_decay_rate = np.array([0.1], dtype=np.float32)

        generate = True
        policy_extraction = True
        if mode == 2:
            generate = False
        elif mode == 3:
            generate = False
            policy_extraction = False

        # Result parameters
        num_sims = len(learning_rate) * len(discount_rate) * len(exploration_rate)
        ts_rewards, ts_steps, ts_successes = [], [], []
        q_tables = np.zeros((num_sims, num_epochs, WIDTH, HEIGHT, WIDTH, HEIGHT, action_space_size))
        # q_tables = np.zeros((num_sims, num_epochs, WIDTH, HEIGHT, action_space_size))

        print("\n# Epochs: ", num_epochs, "\n# Experiments per epoch: ", num_sims, "\n# Episodes per experiment: ", num_episodes)
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
        # if mode == 3:
        #     _, env.grid.shape = extract_values(policy_extraction, load_path, None, env)
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
            ts = 0
            epoch = 0
            for lr_i in np.arange(len(learning_rate)):
                for dr_i in np.arange(len(discount_rate)):
                    for er_i in np.arange(len(exploration_rate)):
                        print("Training session: ", ts)
                        rewards, steps, = [], []
                        for epoch in np.arange(num_epochs):
                            print("Epoch: ", epoch)
                            # Reinitialize some variables
                            ep_exploration_rate = np.copy(exploration_rate)
                            q_table = np.zeros((WIDTH, HEIGHT, WIDTH, HEIGHT, action_space_size))
                            # q_table = np.zeros((WIDTH, HEIGHT, action_space_size))

                            cntr = 0

                            # Q-learning algorithm
                            for episode in range(num_episodes):
                                if episode % 10000 == 0: print("Episode: ", episode)
                                
                                # Initialize new episode params
                                state = QL.reset(env, generate)
                                done = False
                                episode_reward = 0
                                reward = 0

                                for step in range(max_steps_per_episode):
                                    # Exploration-exploitation trade-off
                                    exploration_rate_threshold = random.uniform(0, 1)

                                    # collision checker
                                    surroundings = [False]*(action_space_size) # right, left, up, down

                                    right_is_boundary = env.pos.x == WIDTH - 1
                                    left_is_boundary = env.pos.x == 0
                                    top_is_boundary = env.pos.y == 0
                                    bottom_is_boundary = env.pos.y == HEIGHT - 1

                                    # check if boundary (might not be necessary)
                                    if right_is_boundary:
                                        surroundings[0] = True
                                    if left_is_boundary:
                                        surroundings[1] = True
                                    if top_is_boundary:
                                        surroundings[2] = True
                                    if bottom_is_boundary:
                                        surroundings[3] = True
                                    
                                    temp_actions = []
                                    for i in range(action_space_size):
                                        if not surroundings[i]: temp_actions.append(q_table[tuple(state)][i])
                                        else: temp_actions.append(float("-inf"))

                                    if exploration_rate_threshold > ep_exploration_rate[er_i]:
                                        action = np.argmax(temp_actions)
                                    else:
                                        exclude = [i for i,e in enumerate(temp_actions) if e == float("-inf")]
                                        action = np.random.choice([i for i in range(action_space_size) if i not in exclude])
                                        # action = random.randint(0, action_space_actual_size)

                                    # Take new action
                                    new_state, reward, done, info = QL.step(env, action, pos_reward)

                                    # Update Q-table
                                    q_table[tuple(state+[action])] = q_table[tuple(state+[action])] * (1 - learning_rate[lr_i]) + \
                                        learning_rate[lr_i] * (reward + discount_rate[dr_i] * np.max(q_table[tuple(new_state)][:]))

                                    # Set new state
                                    state = new_state.copy()
                                    
                                    # Add new reward  
                                    episode_reward += reward
                                    
                                    # for debugging
                                    if show_plot:
                                        plt.cla()
                                        PR.print_trajectories(ax, save_path, ts, env, action, reward, done)

                                    if done:
                                        cntr += 1
                                        break

                                # add episode rewards/steps to rewards/steps lists
                                rewards.append(episode_reward)
                                steps.append(step)

                                # Exploration rate decay (Exponential decay)
                                ep_exploration_rate[er_i] = min_exploration_rate[er_i] + \
                                    (max_exploration_rate[er_i] - min_exploration_rate[er_i]) * np.exp(-exploration_decay_rate[er_i]*episode)

                                # calculate average rewards over last 100 episodes (only for display purposes)
                                avg_reward = np.mean(rewards[-print_interval:])
                                avg_steps = np.mean(steps[-print_interval:])

                                # display progress
                                if episode % print_interval == 0 or episode == num_episodes-1 and episode != 0:
                                    percentage = float(cntr)/float(print_interval)*100.0
                                    print('episode= ', episode,
                                            ',reward= %.2f,' % episode_reward,
                                            'average_reward= %.2f,' % avg_reward,
                                            'average_steps= %.2f,' % avg_steps,
                                            'success= %.4f' % (percentage))
                                    cntr = 0

                            # save q table                                    
                            q_tables[ts, epoch] = q_table

                        ts += 1
                        # save rewards in training session rewards
                        ts_rewards.append(rewards)
                        ts_steps.append(steps)

                        training_time = time.time() - training_time
                        print("Time to train policy: %sm %ss" %(divmod(training_time, 60)))

                        # save reward lists and plot learning curve
                        filename = 'learning_cruve%s.png' %(str(ts))
                        filename = os.path.join(save_path, filename)
                        plot_learning_curve(1, "Q-learning", ts_rewards, filename, \
                            learning_rate, discount_rate, exploration_rate, \
                                1, 0, 0, 0.05, \
                                    max_steps_per_episode, training_time)
                        
                        file_name = "ts_rewards%s.json" %(str(ts))
                        file_name = os.path.join(save_path, file_name)
                        write_json(ts_rewards, file_name)

                        file_name = "ts_steps%s.json" %(str(ts))
                        file_name = os.path.join(save_path, file_name)
                        write_json(ts_steps, file_name)

            for i in range(0, q_tables.shape[0]):
                for j in range(0, q_tables.shape[1]):
                    print("\nTesting policy %s epoch %s:" % (i, j))
                    nb_success = 0
                    # Display percentage successes
                    for episode in range(1000):
                        if episode % 1000 == 0: print("Episode: ", episode)
                        # initialize new episode params
                        state = QL.reset(env, generate)
                        #print(env.grid)
                        #print(env.grid)
                        done = False
                        for step in range(max_steps_per_episode):   
                            # Show current state of environment on screen
                            # Choose action with highest Q-value for current state (Greedy policy) 
                            # collision checker
                            surroundings = [False]*(action_space_size) # right, left, up, down

                            right_is_boundary = env.pos.x == WIDTH - 1
                            left_is_boundary = env.pos.x == 0
                            top_is_boundary = env.pos.y == 0
                            bottom_is_boundary = env.pos.y == HEIGHT - 1

                            # check if boundary (might not be necessary)
                            if right_is_boundary:
                                surroundings[0] = True
                            if left_is_boundary:
                                surroundings[1] = True
                            if top_is_boundary:
                                surroundings[2] = True
                            if bottom_is_boundary:
                                surroundings[3] = True
                            
                            temp_actions = []
                            for a in range(action_space_size):
                                if not surroundings[a]: temp_actions.append(q_tables[tuple([i,j] + state)][a])
                                else: temp_actions.append(float("-inf"))   
                            # Take new action
                            action = np.argmax(temp_actions)
                            new_state, reward, done, info = QL.step(env, action, pos_reward)

                            # Set new state
                            state = new_state.copy()
                            
                            if show_plot:
                                plt.cla()
                                PR.print_trajectories(ax, save_path, ts, env, action, reward, done)

                            if done:   
                                nb_success += 1
                                break

                            

                    # Let's check our success rate!
                    print ("Success rate of policy %s = %s %%" % (i, nb_success/100))
            
            winsound.Beep(1000, 500)

            # debug_q2 = input("See optimal policy?\nY/N?")
            # debug_flag2 = str(debug_q2)

#         if mode == 2 or debug_flag2 == 'Y' or debug_flag2 == 'y':
#             policy_extraction = True
#             debug_q3 = input("Policy number?")
#             policy = int(debug_q3)
#             debug_q3 = input("Epoch number?")
#             epoch = int(debug_q3)
#             if mode == 2: correct_path = load_path
#             else: correct_path = save_path
#             q_table_list, env_shape = extract_values(policy_extraction, load_path, policy, epoch, env)

#             q_table = np.array(q_table_list)
#             env.grid = np.empty((int(env_shape[0]), int(env_shape[1])))

#             if mode == 2:
#                 print("\nTesting policy %s:" % (policy))
#                 nb_success = 0
#                 # Display percentage successes
#                 for episode in range(10000):
#                     if episode % 1000 == 0: print("Episode: ", episode)
#                     # initialize new episode params
#                     state = QL.reset(env, generate)
#                     #print(env.grid)
#                     #print(env.grid)
#                     done = False
#                     for step in range(max_steps_per_episode):   
#                         # Show current state of environment on screen
#                         # Choose action with highest Q-value for current state (Greedy policy)     
#                         # Take new action
#                         action = np.argmax(q_table[state][:])
#                         new_state, reward, done, info = QL.step(env, action, pos_reward)
                        
#                         if done:   
#                             nb_success += 1
#                             break

#                         # Set new state
#                         state = new_state.copy()

#                 # Let's check our success rate!
#                 print ("Success rate of policy %s = %s %%" % (policy, nb_success/100))

#             # print("\nTrajectories of policy %s epoch %s:" %(policy, epoch))
#             # test_tab = [None] * (env.grid.shape[1]*env.grid.shape[0])
#             # for s in range(env.grid.shape[0]*env.grid.shape[1]):
#             #     a = np.argmax(q_table[s,:])
#             #     if a == Direction.RIGHT.value:
#             #         test_tab[s] = ">"
#             #     elif a == Direction.LEFT.value: 
#             #         test_tab[s] = "<"
#             #     elif a == Direction.UP.value: 
#             #         test_tab[s] = "^"
#             #     elif a == Direction.DOWN.value: 
#             #         test_tab[s] = "v"
        
#             # print(np.reshape(test_tab, (env.grid.shape[0], env.grid.shape[1])))

#             state = QL.reset(env, generate)
#             env.grid[env.pos.y, env.pos.x] = States.UNEXP.value
#             env.prev_pos = env.starting_pos
#             env.pos = env.prev_pos
#             env.grid[env.pos.y, env.pos.x] = States.ROBOT.value
#             done = False

#             for step in range(max_steps_per_episode):
#                 if mode: all_grids.append(env.grid.copy())       
#                 # Show current state of environment on screen
#                 # Choose action with highest Q-value for current state (Greedy policy)     
#                 # Take new action
#                 action = np.argmax(q_table[state][:]) 
#                 new_state, reward, done, info = QL.step(env, action, pos_reward)
                
#                 if done:
#                     if mode: all_grids.append(env.grid.copy())
#                     break

#                 # Set new state
#                 state = new_state.copy()

                   
#             for i in np.arange(len(all_grids)):
#                 PR = print_results(all_grids[i], env.grid.shape[0], env.grid.shape[1])
#                 PR.print_graph(i)
                
#                 file_name = "plot-%s.png" %(i)
#                 plt.savefig(os.path.join(save_path, file_name))
#                 plt.close()

#         else:   
#             if len(os.listdir(save_path)):
#                 try:
#                     shutil.rmtree(save_path)
#                 except OSError as e:
#                     print("Tried to delete folder that doesn't exist.")


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
            # Clear all visited blocks
            env.grid.fill(0)
            env.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)

            # Setup agent
            # Set new starting pos
            indices = np.argwhere(env.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            env.starting_pos = Point(indices[0,1], indices[0,0])
            env.pos = env.starting_pos
            env.prev_pos = env.pos
            env.grid[env.pos.y, env.pos.x] = States.ROBOT.value

            # Setup goal
            # Set new goal pos
            indices = np.argwhere(env.grid == States.UNEXP.value)
            np.random.shuffle(indices)
            env.goal = Point(indices[0,1], indices[0,0])
            env.grid[env.goal.y, env.goal.x] = States.GOAL.value
        
        env.direction = (Direction.RIGHT).value
        env.exploration_grid = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
        env.exploration_grid[env.pos.y, env.pos.x] = True
                
        self.score = 0
        self.frame_iteration = 0

        selected = self.get_closest_unexplored(env)
        state = self.get_state(env, selected)

        return state
        
    def step(self, env, action, pos_reward):
        self.explored = False
        #print(self.pos, self.prev_pos, action)
        #print(self.grid,"\n")
        self.frame_iteration += 1
        self.p_reward = pos_reward
        self.score = 0
        # 2. Do action
        self._move(env, action) # update the robot
            
        # 3. Update score and get state
        self.score -= 0.05
        selected = self.get_closest_unexplored(env)
        # if selected != None:
        #     if env.pos.x == selected.x and env.pos.y == selected.y:
        #         self.score += self.p_reward
        if self.explored:
            self.score += self.p_reward

        game_over = False

        state = self.get_state(env, selected)
        
        reward = self.score

        # 4. Update environment
        self._update_env(env)

        # 5. Check exit condition
        if (env.exploration_grid == True).all():
            # self.score += self.p_reward
            reward = self.score
            game_over = True
            return state, reward, game_over, self.score

        # 6. return game over and score
        return state, reward, game_over, self.score

    def get_distance(self, end, start):
        return abs(start.x - end.x) + abs(start.y - end.y)
    
    def get_closest_unexplored(self, env):
        distances = {}
        temp_exploration_grid = env.exploration_grid.copy()
        temp_exploration_grid[env.pos.y, env.pos.x] = True
        
        # gets the distance to all unvisited blocks
        for y in range(env.grid.shape[0]):
            for x in range(env.grid.shape[1]):
                if temp_exploration_grid[y,x] == False:
                    distance = self.get_distance(Point(x,y), env.pos)
                    distances[Point(x,y)] = distance
        
        # checks if cell reachable
        if not distances:
            return None
        else:
            return min(distances, key=distances.get)

    def get_state(self, env, selected):
        # return env.pos.y*env.grid.shape[1] + env.pos.x
        #  return [env.pos.x, env.pos.y]
        if selected == None:
            return [env.pos.x, env.pos.y, 0, 0]
        else:
            return [env.pos.x, env.pos.y, selected.x, selected.y]
    
    def _is_collision(self, env, pt=None):
        if pt is None:
            pt = env.pos
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
            env.grid[env.prev_pos.y,env.prev_pos.x] = States.EXP.value
            env.grid[env.pos.y,env.pos.x] = States.ROBOT.value
        else:
            env.prev_pos = env.pos
            env.pos = Point(x,y)
            if env.grid[env.pos.y, env.pos.x] == States.UNEXP.value:
                self.explored = True
            env.exploration_grid[env.pos.y, env.pos.x] = True
            env.grid[env.prev_pos.y,env.prev_pos.x] = States.EXP.value
            env.grid[env.pos.y,env.pos.x] = States.ROBOT.value

# def moving_avarage_smoothing(X,k):
# 	S = np.zeros(X.shape[0])
# 	for t in range(X.shape[0]):
# 		if t < k:
# 			S[t] = np.mean(X[:t+1])
# 		else:
# 			S[t] = np.sum(X[t-k:t])/k
# 	return S

# # Calculates average rewards and steps
# def calc_avg(rewards, steps, num_epochs, num_sims):
#     avg_rewards = np.sum(np.array(rewards), axis=0)
#     avg_steps = np.sum(np.array(steps), axis=0)

#     avg_rewards = np.divide(avg_rewards, num_epochs)
#     avg_steps = np.divide(avg_steps, num_epochs)

#     mov_avg_rewards = np.empty(avg_rewards.shape)
#     mov_avg_steps = np.empty(avg_steps.shape)

#     for i in range(0, num_sims):
#         for j in range(0, num_epochs):
#             mov_avg_rewards[i] = moving_avarage_smoothing(avg_rewards[i,j], 100)
#             mov_avg_steps[i] = moving_avarage_smoothing(avg_steps[i,j], 100)

#     # return mov_avg_rewards.tolist(), mov_avg_steps.tolist()
#     return avg_rewards.tolist(), avg_steps.tolist()

# def extract_values(policy_extraction, correct_path, policy, epoch, env):
#     f = open(os.path.join(correct_path,"saved_data.txt"), "r")
#     lines = f.readlines()

#     for line in lines:
#         cur_line = []
#         for char in line:
#             if char.isdigit():
#                 cur_line.append(char)

#     if policy_extraction:
#         file_name = "policy" + str(policy) + "_" + str(epoch) + ".txt"
#         return np.loadtxt(os.path.join(correct_path, file_name)), cur_line
#     else:
#         return None, cur_line

                    