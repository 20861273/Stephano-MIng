# Notes:
# Last at line 186

import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env, write_json, read_hp_json, read_json
from gym import wrappers
from dqn_environment import Environment, HEIGHT, WIDTH, Point, States
from datetime import datetime
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt
import time

def dqn(training_sessions, episodes, discount_rate, epsilon,
        batch_size, n_actions, eps_min, eps_dec, input_dims, 
        learning_rate, positive_reward, negative_reward, max_steps, i_exp,
        models_path, env_name, load_checkpoint):
    start_time = time.time()
    ts_rewards = []
    for i_ts in range(training_sessions):
        print("Experience: %s, Epoch: %s, Discount rate: %s, Learning rate: %s, Epsilon: %s" %(str(i_exp), str(i_ts), str(discount_rate), str(learning_rate), str(epsilon)))
        rewards, steps = [], []
        env = Environment(positive_reward, negative_reward)
        agent = DQNAgent(gamma=discount_rate, epsilon=epsilon, eps_min=eps_min, eps_dec=eps_dec, lr=learning_rate,
                     n_actions=n_actions, input_dims=input_dims, mem_size=50000,
                     batch_size=batch_size, replace=1000,
                     algo='DQNAgent', env_name=env_name, chkpt_dir=models_path)
        if load_checkpoint:
            agent.load_models()

        cntr = 0
        best_reward = -np.inf
        # last_start = -1 # iterative start
        last_start = None # random start
        for i_episode in range(episodes):
            episode_reward = 0
            done = False
            observation, last_start = env.reset(last_start)
            for step in range(max_steps):
                action = agent.choose_action(observation)
                # observation_, reward, done, info, cntr = env.step(action, cntr)
                observation_, reward, done, info = env.step(action)
                episode_reward += reward
                if not load_checkpoint:
                    agent.store_transition(observation, action,
                                        reward, observation_, done)
                    agent.learn()
                observation = observation_
                if done:
                    cntr += 1
                    break
            rewards.append(episode_reward)
            steps.append(step)

            avg_reward = np.mean(rewards[-100:])
            avg_steps = np.mean(steps[-100:])
            if avg_reward > best_reward:
                if not load_checkpoint:
                    agent.save_models()
                    file_name = "experience%s.pth" %(str(i_exp))
                    file_name = os.path.join(save_path, file_name)
                    T.save(agent.q_eval.state_dict(),file_name)
                best_reward = avg_reward

            if i_episode % 1000==0 or i_episode == episodes-1:
                print('episode ', i_episode, 'reward %.2f' % episode_reward,
                        'average reward %.2f' % avg_reward,
                        'average steps %.2f' % avg_steps,
                        'dones %d' % cntr)
                cntr = 0
            
        ts_rewards.append(rewards)
    avg_rewards = [0]*len(ts_rewards[0])
    for i_ts in range(len(ts_rewards[0])):
        s = sum(ts_rewards[j][i_ts] for j in range(len(ts_rewards)))
        avg_rewards[i_ts] = s / len(ts_rewards)
    x = [i+1 for i in range(episodes)]
    
    end_time = time.time()
    total_time = end_time - start_time

    filename = 'learning_cruve%s.png' %(str(i_exp))
    filename = os.path.join(save_path, filename)
    plot_learning_curve(x, avg_rewards, filename, learning_rate, discount_rate, epsilon, positive_reward, negative_reward, max_steps, total_time)
    # file_name = "experience%s.pth" %(str(i_exp))
    # file_name = os.path.join(save_path, file_name)
    # T.save(agent.q_eval.state_dict(),file_name)
    file_name = "rewards%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(avg_rewards, file_name)
    spawning = "iterative"
    if last_start == None: spawning = "random"
    hp = "%s,%s,%s,%s,%s,%s,%s,%s" %(str(training_sessions), str(learning_rate),str(discount_rate), str(epsilon), str(positive_reward), str(negative_reward), str(max_steps), spawning)
    file_name = "hyperparameters%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(hp, file_name)

if __name__ == '__main__':
    # for on-policy runs
    off_policy = False
    policy_num = 1
    testing_iterations = 1000

    # initialize hyperparameters
    learning_rate = [0.001]
    discount_rate = [0.5, 0.9]
    epsilon = [0.01]
    eps_min = [0.01]

    batch_size = 64

    n_actions = 4
    input_dims = [HEIGHT*WIDTH]

    training_sessions = 1
    episodes = 100000
    positive_rewards = [2]
    negative_rewards = [0]
    max_steps = [200]

    num_experiences = len(learning_rate) * len(discount_rate) * len(epsilon) * len(positive_rewards) * len(negative_rewards) * len(max_steps) * training_sessions

    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'SAR')
    PATH = os.path.join(PATH, 'Results')
    PATH = os.path.join(PATH, 'DQN')
    load_path = os.path.join(PATH, 'Saved_data')
    if not os.path.exists(load_path): os.makedirs(load_path)
    date_and_time = datetime.now()
    save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(save_path): os.makedirs(save_path)
    models_path = os.path.join(save_path, 'models')
    if not os.path.exists(models_path): os.makedirs(models_path)

    env_name = 'SAR%sx%s' %(str(WIDTH), str(HEIGHT))
    load_checkpoint = False

    if off_policy:
        print("Number of training sessoins: ", num_experiences)
        i_exp = 0
        for pr_i in positive_rewards:
            for nr_i in negative_rewards:
                for ms_i in max_steps:
                    for lr_i in learning_rate:
                        for dr_i in discount_rate:
                            for er_i in epsilon:
                                rewards = dqn(training_sessions,
                                                episodes,
                                                dr_i,
                                                er_i,
                                                batch_size,
                                                n_actions,
                                                er_i,
                                                er_i,
                                                input_dims,
                                                lr_i,
                                                pr_i,
                                                nr_i,
                                                ms_i,
                                                i_exp,
                                                models_path,
                                                env_name,
                                                load_checkpoint)
                                i_exp += 1
    else:
        file_name = "hyperparameters%s.json" %(str(policy_num))
        ts, lr, dr, er, pr, nr, ms = read_hp_json(load_path, file_name, policy_num)
        # saved hyperparameters should be used as input
        agent = DQNAgent(gamma=dr, epsilon=0, eps_min=0, eps_dec=0, lr=lr,
                     n_actions=n_actions, input_dims=input_dims, mem_size=50000,
                     batch_size=batch_size, replace=1000,
                     algo='DQNAgent', env_name=env_name, chkpt_dir=models_path)
        file_name = "experience%s.pth" %(str(policy_num))
        file_name = os.path.join(load_path, file_name)
        agent.q_eval.load_state_dict(T.load(file_name))
        agent.q_eval.eval()
        agent.q_next.load_state_dict(T.load(file_name))
        agent.q_next.eval()
        env = Environment(pr, nr)

        temp_step_grid = np.empty(env.grid.shape, dtype=object)
        for i in np.ndindex(temp_step_grid.shape): temp_step_grid[i] = []
        temp_step_grid = temp_step_grid.tolist()
        step_grid = np.zeros(env.grid.shape)
        cnt = 0
        trajectories = np.empty(env.grid.shape, dtype=object)
        for i in np.ndindex(trajectories.shape): trajectories[i] = []
        trajectories = trajectories.tolist()
        for y in range(env.grid.shape[0]):
            for x in range(env.grid.shape[1]):
                # insert for loop here for monte carlo testing (not here because this is for trajectory plotting)
                # check spawning. spawns on same cell some times
                print("Testing: x=%d, y=%d" %(x, y))
                for i in range(testing_iterations):
                    observation, last_start = env.reset(last_start=None)
                    env.grid[env.pos.y, env.pos.x] = States.UNEXP.value
                    
                    while env.goal == Point(x,y):
                        observation, last_start = env.reset(last_start=None)
                        env.grid[env.pos.y, env.pos.x] = States.UNEXP.value
                    
                    env.pos = Point(x,y)
                    env.prev_pos = Point(x,y)
                    env.starting_pos = Point(x,y)
                    env.grid[env.pos.y, env.pos.x] = States.ROBOT.value

                    trajectory = [env.goal]

                    done = False
                    actions = []

                    # state_observation = env.get_state()
                    # goal_observation = env.get_goal_state()
                    # observation = np.append(state_observation, goal_observation, axis=0)
                    observation = env.get_state_unex()
                    # observation = state_observation.copy()
                
                    for step in range(int(ms)):
                        action = agent.choose_action(observation)
                        actions.append(action)
                        observation_, reward, done, _ = env.step(action)
                        observation = observation_
                        trajectory.append((env.prev_pos, action))
                        if done:
                            trajectories[y][x] = trajectory
                            cnt += 1
                            break
                    temp_step_grid[y][x].append(step)
                step_grid[y,x] = sum(temp_step_grid[y][x])/len(temp_step_grid[y][x])
                # if done: break
            # if done: break
        print("Percentage success: %d / %d x 100 = %.2f" %(cnt, HEIGHT*WIDTH*testing_iterations, cnt/((HEIGHT*WIDTH*testing_iterations))*100))
        print(step_grid)
        file_name = "policy%s_results.txt" %(str(policy_num))
        file_name = os.path.join(save_path, file_name)
        np.savetxt(file_name, step_grid, fmt="%d")
        cnt = 0
        for i, traj in enumerate(trajectories):
            for j, t in enumerate(traj):
                PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])
                PR.print_row(t, save_path, i*env.grid.shape[0]+j, env)
