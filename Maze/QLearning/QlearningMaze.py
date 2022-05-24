import numpy as np
import random
import time

from ql_maze import MazeAI, Direction, Point

# Creating The Environment
env = MazeAI()

# Creating The Q-Table
action_space_size = len(Direction)
state_space_size = len(env.grid[0])*len(env.grid[1])


q_table = np.zeros((state_space_size, action_space_size))

# Initializing Q-Learning Parameters
num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

# Training Loop
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):         
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
            
        else:
            action = random.randint(0, action_space_size-1)

        # Take new action
        new_state, reward, done, info = env.step(action)

        # Update Q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        # Set new state
        state = new_state
        # Add new reward  
        rewards_current_episode += reward

        #print(action, reward)
        #print(env.grid)
        
        #print(rewards_current_episode)
        if done == True:
            break

        # Exploration rate decay 
        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode) 
        # Add current episode reward to total rewards list
        rewards_all_episodes.append(rewards_current_episode)





print("\n\n********Q-table********\n")
print(q_table)

nb_success = 0
# Display percentage successes
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()
    done = False
    for step in range(max_steps_per_episode):        
        # Show current state of environment on screen
        # Choose action with highest Q-value for current state       
        # Take new action
        action = np.argmax(q_table[state,:])        
        new_state, reward, done, info = env.step(action)
        
        if done:
            if reward > 0:
                nb_success += 1
            break       

        # Set new state
        state = new_state

# Let's check our success rate!
print (f"Success rate = {nb_success/num_episodes*100}%")

# Display single episode
for episode in range(1):
    # initialize new episode params
    state = env.reset()
    done = False
    for step in range(max_steps_per_episode):        
        # Show current state of environment on screen
        # Choose action with highest Q-value for current state       
        # Take new action
        print("Step:" , step )
        action = np.argmax(q_table[state,:])        
        new_state, reward, done, info = env.step(action)

        print(env.grid)
        
        if done:
            if reward > 0:
                nb_success += 1
            break       

        # Set new state
        state = new_state
