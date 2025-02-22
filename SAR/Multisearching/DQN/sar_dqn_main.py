# Notes:
# Last at line 

import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, write_json, read_hp_json, read_json
from dqn_environment import Environment, HEIGHT, WIDTH, Point, States, Direction
from datetime import datetime
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def dqn(nr, training_sessions, episodes, discount_rate, epsilon,
        batch_size, n_actions, eps_min, eps_dec, input_dims, 
        learning_rate, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, max_steps, i_exp,
        models_path, env_name, load_checkpoint, replace):
    start_time = time.time()
    ts_rewards = []
    for i_ts in range(training_sessions):
        print("Experience: %s, Epoch: %s,\n\
            Discount rate: %s, Learning rate: %s, Epsilon: %s\n\
            Positive goal reward: %s, Negative collision reward:%s\n\
            Positive exploration reward: %s, Negative step reward:%s\n\
            Max steps: %s, Replace: %s"\
                 %(str(i_exp), str(i_ts), \
                    str(discount_rate), str(learning_rate), str(epsilon),\
                    str(positive_reward), str(negative_reward), str(positive_exploration_reward), str(negative_step_reward),\
                    str(max_steps), str(replace)))
        rewards, steps = [], []
        env = Environment(nr, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward)
        agent = DQNAgent(nr, gamma=discount_rate, epsilon=epsilon, eps_min=eps_min, eps_dec=eps_dec, lr=learning_rate,
                     n_actions=n_actions, input_dims=input_dims, mem_size=50000,
                     batch_size=batch_size, replace=replace,
                     algo='DQNAgent', env_name=env_name, chkpt_dir=models_path)
        if load_checkpoint:
            agent.load_models()

        cntr = 0
        best_reward = -np.inf
        for i_episode in range(episodes):
            episode_reward = 0
            done = False
            observation = env.reset()
            breakpoint
            for step in range(max_steps):
                for i_r in range(0,nr):
                    action = agent.choose_action(observation)
                    observation_, reward, done, info = env.step(action, i_r)
                    episode_reward += reward

                    if not load_checkpoint:
                        agent.store_transition(observation, action,
                                            reward, observation_, done)
                        agent.learn()

                    observation = observation_

                    if done:
                        cntr += 1
                        break
                if done: break
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

            if i_episode % 50==0 or i_episode == episodes-1 and i_episode != 0:
                print('episode= ', i_episode,
                        ',reward= %.2f,' % episode_reward,
                        'average_reward= %.2f,' % avg_reward,
                        'average_steps= %.2f,' % avg_steps,
                        'success= %.4f' % (float(cntr)/50.0*100.0))
                cntr = 0
            
        ts_rewards.append(rewards)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)

    filename = 'learning_cruve%s.png' %(str(i_exp))
    filename = os.path.join(save_path, filename)
    plot_learning_curve(ts_rewards, filename, \
        learning_rate, discount_rate, epsilon, \
            positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, \
                max_steps, total_time)
    # file_name = "experience%s.pth" %(str(i_exp))
    # file_name = os.path.join(save_path, file_name)
    # T.save(agent.q_eval.state_dict(),file_name)
    file_name = "rewards%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(ts_rewards, file_name)
    spawning = "random"
    # if last_start == None: spawning = ""
    hp = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" %(\
        str(training_sessions), \
        str(learning_rate),\
        str(discount_rate), \
        str(epsilon), \
        str(positive_reward), \
        str(negative_reward), \
        str(positive_exploration_reward), \
        str(negative_step_reward), \
        str(max_steps), \
        observation_state,\
        termination_state, \
        spawning, \
        str(replace))
    file_name = "hyperparameters%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(hp, file_name)

if __name__ == '__main__':
    # Testing: for on-policy runs
    off_policy = False
    show_plot = False
    policy_num = [0]
    testing_iterations = 10000

    nr = 2

    # states:
    # observation states: "position", "position_explored", "image"
    # termination states: "goal", "covered"
    observation_state = "image"
    termination_state = "goal"

    # initialize hyperparameters
    learning_rate = [0.0001]
    discount_rate = [0.9]
    epsilon = [0.01]
    eps_min = [0.01]

    # NN
    batch_size = 64

    n_actions = 4
    input_dims = []
    if observation_state == "position":
        input_dims = [HEIGHT*WIDTH]
    elif observation_state == "position_explored":
        input_dims = [HEIGHT*WIDTH*2]
    elif observation_state == "image":
        input_dims = (3,HEIGHT, WIDTH)

    replace = 1000

    training_sessions = 3
    episodes = 20000
    positive_rewards = [1]
    positive_exploration_rewards = [0]
    negative_rewards = [0.1]
    negative_step_rewards = [0.01]
    max_steps = [200]

    num_experiences =     len(learning_rate) \
                        * len(discount_rate) \
                        * len(epsilon) \
                        * len(positive_rewards) \
                        * len(negative_rewards) \
                        * len(positive_exploration_rewards) \
                        * len(negative_step_rewards) \
                        * len(max_steps) \
                        * training_sessions

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
                for per_i in positive_exploration_rewards:
                    for nsr_i in negative_step_rewards:
                        for ms_i in max_steps:
                            for lr_i in learning_rate:
                                for dr_i in discount_rate:
                                    for er_i in epsilon:
                                        rewards = dqn(nr,
                                                    training_sessions,
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
                                                    per_i,
                                                    nsr_i,
                                                    ms_i,
                                                    i_exp,
                                                    models_path,
                                                    env_name,
                                                    load_checkpoint,
                                                    replace)
                                        i_exp += 1
    else:
        for policy in policy_num:
            debug = True
            print("Testing policy %d:" %(policy))
            file_name = "hyperparameters%s.json" %(str(policy))
            ts, lr, dr, er, pr, ner, per, nsr, ms, r = read_hp_json(load_path, file_name, policy)
            
            agent = DQNAgent(nr, gamma=dr, epsilon=0, eps_min=0, eps_dec=0, lr=lr,
                        n_actions=n_actions, input_dims=input_dims, mem_size=50000,
                        batch_size=batch_size, replace=r,
                        algo='DQNAgent', env_name=env_name, chkpt_dir=models_path)
            file_name = "experience%s.pth" %(str(policy))
            file_name = os.path.join(load_path, file_name)
            agent.q_eval.load_state_dict(T.load(file_name))
            agent.q_eval.eval()
            agent.q_next.load_state_dict(T.load(file_name))
            agent.q_next.eval()
            env = Environment(nr, pr, ner, per, nsr)

            PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])

            trajs = []
            steps = []
            cnt = 0
            trajectories = []
            for i in range(0, testing_iterations):
                if i % 1000 == 0 and i != 0:
                    print("%d: %.2f %%, %.2f steps" %(int(i), float(cnt)/float(i)*100, np.mean(np.array(steps))))
                observation = env.reset()

                trajectory = []

                done = False
                actions = []

                if show_plot:
                    PR.print_trajectories(save_path, policy, env)
                    plt.close()
            
                for step in range(int(ms)):
                    actions = []
                    for i_r in range(0,nr):
                        action = agent.choose_action(observation)
                        actions.append(action)
                        trajectory.append((env.pos[i_r], action, i_r))
                        observation_, reward, done, info = env.step(action, i_r)

                        if not load_checkpoint:
                            agent.store_transition(observation, action,
                                                reward, observation_, done)
                            agent.learn()

                        observation = observation_

                        if done:
                            if show_plot:
                                PR.print_trajectories(save_path, policy, env, actions)
                                plt.close()
                            trajectories.append(trajectory)
                            cnt += 1
                            break
                    if done: break
                    if show_plot:
                        PR.print_trajectories(save_path, policy, env, actions)
                        plt.close()
                # if step == int(ms)-1 and not done:
                #     trajectories.append(trajectory)
                steps.append(step)

            p = cnt/(testing_iterations)*100
            print("Percentage success: %d / %d x 100 = %.2f %%" %(cnt, testing_iterations, p))
