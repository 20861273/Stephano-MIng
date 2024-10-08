# Notes:
# Last at line 

import numpy as np
from dqn_agent_simultaneous import DQNAgent
from dqn_centralized_training import dqn
from utils import plot_learning_curve, write_json, read_hp_json, read_json, save_hp, read_checkpoint_hp_json
from dqn_environment_simultaneous import Environment, HEIGHT, WIDTH, Point, States, Direction
from datetime import datetime
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # Testing: for on-policy runs
    off_policy = False
    show_plot = True
    policy_num = [0,1,2,3]
    testing_iterations = 1000

    load_checkpoint = False
    n_experiences = 4
    start_up_exp = 1

    nr = 2

    # states:
    # observation states: "position", "position_explored", "image"
    # termination states: "goal", "covered"
    observation_state = "image"
    termination_state = "goal"

    # initialize hyperparameters
    learning_rate = [0.0001]
    discount_rate = [0.9,0.99]
    epsilon = [1, 1]
    eps_min = [0.01, 0.05]
    eps_dec = [0.001, 0.001]

    # NN
    batch_size = 64
    mem_size = 50000
    replace = 1000
    c_dims = [16, 32]
    k_size = [2, 2]
    s_size = [1, 1]
    fc_dims = [32]

    n_actions = 4
    input_dims = []
    if observation_state == "position":
        input_dims = [HEIGHT*WIDTH]
    elif observation_state == "position_explored":
        input_dims = [HEIGHT*WIDTH*2]
    elif observation_state == "image":
        input_dims = (3,HEIGHT+2, WIDTH+2)

    training_sessions = 10
    episodes = 100000
    positive_rewards = [1]
    positive_exploration_rewards = [0]
    negative_rewards = [1]
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

    env_size = '%sx%s' %(str(WIDTH), str(HEIGHT))
    

    save_hp(save_path, nr, training_sessions, episodes,
            positive_rewards, negative_rewards, positive_exploration_rewards, negative_step_rewards,
            max_steps, learning_rate, discount_rate, epsilon,
            n_actions,
            c_dims, k_size, s_size, fc_dims,
            batch_size, mem_size, replace, env_size)

    if load_checkpoint:
        # initialize hyperparameters
        learning_rate = []
        discount_rate = []
        epsilon = []
        eps_min = []

        # NN
        batch_size = 0
        mem_size = 0
        replace = 0
        c_dims = []
        k_size = []
        s_size = []
        fc_dims = []

        n_actions = 0

        training_sessions = 0
        episodes = 0
        positive_rewards = []
        positive_exploration_rewards = []
        negative_rewards = []
        negative_step_rewards = []
        max_steps = []
        for i in range(n_experiences):
            if i >= start_up_exp:
                file_name = "hyperparameters%s.json" %(str(i))
                file_name = os.path.join(load_path, file_name)
                training_sessions, nr, episodes, lr, dr, er, pr, neg_r, per, nsr, ms, n_actions, c_d, k_s, s_s, fc_d,mem_size, batch_size, replace, env_size = read_hp_json(load_path, file_name)
                learning_rate.append(lr)
                discount_rate.append(dr)
                epsilon.append(er)
                eps_min.append(er)
                positive_rewards.append(pr)
                positive_exploration_rewards.append(per)
                negative_rewards.append(neg_r)
                negative_step_rewards.append(nsr)
                max_steps.append(ms)
                c_dims = c_d.copy()
                k_size = k_s.copy()
                s_size = s_s.copy()
                fc_dims = fc_d.copy()

    if off_policy:
        print("Number of training sessoins: ", num_experiences)
        i_exp = 0
        if load_checkpoint: i_exp = start_up_exp
        for pr_i in positive_rewards:
            for nr_i in negative_rewards:
                for per_i in positive_exploration_rewards:
                    for nsr_i in negative_step_rewards:
                        for ms_i in max_steps:
                            for lr_i in learning_rate:
                                for dr_i in discount_rate:
                                    for er_i in epsilon:
                                        
                                        rewards = dqn(nr, training_sessions, episodes, dr_i, lr_i, er_i, er_i, er_i,
                                                    pr_i, nr_i, per_i, nsr_i, ms_i, i_exp,
                                                    n_actions, input_dims,
                                                    c_dims, k_size, s_size, fc_dims,
                                                    batch_size, mem_size, replace,
                                                    models_path, load_path, save_path, env_size, load_checkpoint, start_up_exp)
                                        i_exp += 1
    else:
        for policy in policy_num:
            debug = True
            print("Testing policy %d:" %(policy))
            file_name = "hyperparameters%s.json" %(str(policy))
            ts, lr, dr, er, pr, ner, per, nsr, ms, n_actions, c_dims, k_size, s_size, fc_dims, mem_size, batch_size, r = read_hp_json(load_path, file_name)
            # int(ts),float(lr), float(dr), float(er), float(pr), float(nr), float(per), float(nsr), int(ms), int(n_actions), c_dims, k_size, s_size, fc_dims,int(mem_size), int(batch_size), int(replace),env_size
            
            # nr, gamma, epsilon, eps_min, eps_dec, lr, n_actions, starting_beta,
            #      input_dims, c_dims, k_size, s_size, fc_dims,
            #      mem_size, batch_size, replace, prioritized=False, algo=None, env_name=None, chkpt_dir='tmp/dqn'

            agent = DQNAgent(nr, dr, er, er, er, lr,
                     n_actions, 0.4, input_dims,
                     c_dims, k_size, s_size, fc_dims,
                     mem_size=mem_size,
                     batch_size=batch_size, replace=replace, prioritized=False,
                     algo='DQNAgent', env_name=env_size, chkpt_dir=models_path)
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
            fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))
            for i in range(0, testing_iterations):
                if i % 1000 == 0 and i != 0:
                    print("%d: %.2f %%, %.2f steps" %(int(i), float(cnt)/float(i)*100, np.mean(np.array(steps))))
                observation = env.reset()

                trajectory = []

                done = False
                

                if show_plot:
                    PR.print_trajectories(ax, save_path, policy, env)
            
                for step in range(int(ms)):
                    actions = []
                    action = [0]*nr
                    for i_r in range(0,nr):
                        action[i_r] = agent.choose_action(observation[i_r])
                    actions.append(action)
                    trajectory.append((env.pos[i_r], action, i_r))
                    observation_, reward, done, info = env.step(action)
                    cnt += info

                    # if not load_checkpoint:
                    #     for i_r in range(0,nr):
                    #         agent.store_transition(observation[i_r], action[i_r],
                    #                             reward, observation_[i_r], done)
                    #         agent.learn()

                    observation = observation_

                    if done:
                        if show_plot:
                            plt.cla()
                            PR.print_trajectories(ax, save_path, policy, env, actions[0])
                        trajectories.append(trajectory)
                        break
                    if show_plot:
                        plt.cla()
                        PR.print_trajectories(ax, save_path, policy, env, actions[0])
                steps.append(step)
                # if step == int(ms)-1 and not done:
                #     trajectories.append(trajectory)

            p = cnt/(testing_iterations)*100
            print("Percentage success: %d / %d x 100 = %.2f %%" %(cnt, testing_iterations, p))
            print("Average steps: %.2f" %(np.mean(np.array(steps))))
