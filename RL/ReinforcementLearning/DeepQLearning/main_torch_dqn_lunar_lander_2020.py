import gym
from simple_dqn_torch_2020 import Agent
from utils import plot_learning_curve, write_json, read_hp_json
import numpy as np
from dqn_environment import Environment, HEIGHT, WIDTH, Point
import time
from datetime import datetime
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt
import json

def dqn(training_sessions,
        episodes,
        discount_rate,
        epsilon,
        batch_size, 
        n_actions, 
        eps_end, 
        input_dims, 
        learning_rate, 
        positive_reward, 
        max_steps, 
        i_exp):
    ts_rewards = []
    for i_ts in range(training_sessions):
        print("Experience: ", i_exp)
        print("Discount rate: %s\nLearning rate: %s\nEpsilon: %s" %(str(discount_rate), str(learning_rate), str(epsilon)))
        print("Epoch: ", i_ts)
        rewards = []
        env = Environment(positive_reward)
        agent = Agent(gamma=discount_rate, epsilon=epsilon, batch_size=batch_size, n_actions=n_actions, eps_end=eps_end,
                input_dims=input_dims, lr=learning_rate)
        cntr = 0
        last_start = -1 # iterative start
        # last_start = None # rnadom start
        for i_episode in range(episodes):
            episode_reward = 0
            done = False
            observation, last_start = env.reset(last_start)
            for step in range(max_steps):
                action = agent.choose_action(observation)
                # observation_, reward, done, info, cntr = env.step(action, cntr)
                observation_, reward, done, info = env.step(action)
                episode_reward += reward
                agent.store_transition(observation, action, reward, 
                                        observation_, done)
                agent.learn()
                observation = observation_
                if done:
                    cntr += 1
                    break
            rewards.append(episode_reward)

            avg_reward = np.mean(rewards[-100:])

            if i_episode % 1000==0:
                print('episode ', i_episode, 'reward %.2f' % episode_reward,
                        'average reward %.2f' % avg_reward,
                        'epsilon %.2f' % agent.epsilon)
                print(cntr)
                cntr = 0
            
        ts_rewards.append(rewards)
    avg_rewards = [0]*len(ts_rewards[0])
    for i_ts in range(len(ts_rewards[0])):
        s = sum(ts_rewards[j][i_ts] for j in range(len(ts_rewards)))
        avg_rewards[i_ts] = s / len(ts_rewards)
    x = [i+1 for i in range(episodes)]
    
    filename = 'learning_cruve%s.png' %(str(i_exp))
    filename = os.path.join(save_path, filename)
    plot_learning_curve(x, avg_rewards, filename, learning_rate, discount_rate, epsilon)
    file_name = "experience%s.pth" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    T.save(agent.Q_eval.state_dict(),file_name)
    file_name = "rewards%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(avg_rewards, file_name)
    spawning = "iterative"
    if last_start == None: spawning = "random"
    hp = "%s,%s,%s,%s,%s,%s" %(str(learning_rate),str(discount_rate), str(epsilon), str(positive_reward), str(max_steps), spawning)
    file_name = "hyperparameters%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(hp, file_name)


if __name__ == '__main__':
    # for on-policy runs
    on_policy = True
    policy_num = 2

    # initialize hyperparameters
    learning_rate = [0.001]
    discount_rate = [0.1, 0.5, 0.9]
    epsilon = [0.03]
    eps_end = [0.03]

    batch_size = 64

    n_actions = 4
    input_dims = [HEIGHT*WIDTH*2]

    training_sessions = 5
    episodes = 30000
    positive_rewards = [1]
    max_steps = [100]

    num_experiences = len(learning_rate) * len(discount_rate) * len(epsilon) * len(positive_rewards) * len(max_steps) * training_sessions

    print("Number of training sessoins: ", num_experiences)

    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'SAR')
    PATH = os.path.join(PATH, 'Results')
    PATH = os.path.join(PATH, 'DQN')
    load_path = os.path.join(PATH, 'Saved_data')
    if not os.path.exists(load_path): os.makedirs(load_path)
    date_and_time = datetime.now()
    save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(save_path): os.makedirs(save_path)

    if on_policy:
        i_exp = 0
        for pr_i in positive_rewards:
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
                                            input_dims,
                                            lr_i,
                                            pr_i,
                                            ms_i,
                                            i_exp)
                            i_exp += 1
        
    else:
        file_name = "hyperparameters%s.json" %(str(policy_num))
        lr, dr, er, pr, ms = read_hp_json(load_path, file_name, policy_num)
        # saved hyperparameters should be used as input
        agent = Agent(gamma=dr, epsilon=er, batch_size=64, n_actions=4, eps_end=er,
                    input_dims=[HEIGHT*WIDTH*2], lr=lr)
        file_name = "experience%s.pth" %(str(policy_num))
        file_name = os.path.join(load_path, file_name)
        agent.Q_eval.load_state_dict(T.load(file_name))
        agent.Q_eval.eval()
        env = Environment(1)

        found = []
        resultss = []
        cnt = 0
        for y in range(env.grid.shape[0]):
            for x in range(env.grid.shape[1]):
                if (x,y) == (0,0): continue
                observation, last_start = env.reset(last_start=None)
                grids = []
                results = []
                
                env.grid[env.pos.y, env.pos.x] = 0
                env.pos = Point(x,y)
                env.prev_pos = Point(x,y)
                env.starting_pos = Point(x,y)
                env.grid[env.pos.y, env.pos.x] = 2

                grids.append(env.grid.copy())
                results.append((env.pos, None))

                done = False
                actions = []
            
                for step in range(max_steps[0]):
                    action = agent.choose_action(observation)
                    actions.append(action)
                    next_state_, reward, done, _ = env.step(action)
                    results.append((env.pos, action))
                    grids.append(env.grid.copy())
                    state = next_state_
                    if done:
                        found.append(grids)
                        resultss.append(results)
                        cnt += 1
                        break
                # if done: break
            # if done: break
        print("Percentage success: ", cnt, HEIGHT*WIDTH, cnt/((HEIGHT*WIDTH)-1)*100)
        cnt = 0
        for j, g in enumerate(found):
            for i, g_i in enumerate(g):
                PR = print_results(g_i, env.grid.shape[0], env.grid.shape[1])
                PR.print_graph(i)
                
                file_name = "plot-%s_%s.png" %(j, i)
                plt.savefig(os.path.join(save_path, file_name))
                plt.close()

