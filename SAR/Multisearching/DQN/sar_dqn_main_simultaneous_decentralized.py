# Notes:
# Last at line 

import numpy as np
from dqn_agent_simultaneous import DQNAgent
from utils_decentralized import plot_learning_curve, write_json, read_hp_json, read_json, save_hp, read_checkpoint_hp_json
from dqn_environment_simultaneous_decentralized import Environment, HEIGHT, WIDTH, Point, States, Direction
from datetime import datetime
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt
import time
from dqn_decentralized_training import dqn
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # Testing: for on-policy runs
    off_policy = False
    show_plot = False
    policy_num = [0]
    testing_iterations = 1000

    load_checkpoint = False
    n_experiences = 3
    start_up_exp = 0

    nr = 2

    # states:
    # observation states: "position", "position_explored", "image"
    # termination states: "goal", "covered"
    observation_state = "image"
    termination_state = "goal"

    # initialize hyperparameters
    learning_rate = [0.0001]
    discount_rate = [0.9]
    epsilon = [[0.01,0.01,0.01]] # epsilon, epsilon min, epsilon dec

    # NN
    batch_size = 64
    mem_size = 50000
    replace = 1000 # test 100 replace
    c_dims = [16, 32]
    k_size = [2, 2]
    s_size = [1, 1]
    fc_dims = [32]

    # PER
    prioritized = True
    starting_beta = 0.5

    n_actions = 4
    input_dims = []
    if observation_state == "position":
        input_dims = [HEIGHT*WIDTH]
    elif observation_state == "position_explored":
        input_dims = [HEIGHT*WIDTH*2]
    elif observation_state == "image":
        input_dims = (3,HEIGHT, WIDTH)

    training_sessions = 1
    episodes = 30000
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

    load_checkpoint_path = os.path.join(PATH, "01-06-2023 11h43m00s")
    if load_checkpoint: save_path = load_checkpoint_path

    env_size = '%sx%s' %(str(WIDTH), str(HEIGHT))
    
    if not load_checkpoint:
        save_hp(save_path, nr, training_sessions, episodes,
                positive_rewards, negative_rewards, positive_exploration_rewards, negative_step_rewards,
                max_steps, learning_rate, discount_rate, epsilon,
                n_actions,
                c_dims, k_size, s_size, fc_dims, prioritized,
                batch_size, mem_size, replace, env_size)

    if load_checkpoint and off_policy:
        # initialize hyperparameters
        learning_rate = []
        discount_rate = []
        epsilon = []
        eps_min = []
        eps_dec = []

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
                file_name = os.path.join(load_checkpoint_path, file_name)
                training_sessions, nr, episodes, lr, dr, er, pr, neg_r, per, nsr, ms, n_actions, c_d, k_s, s_s, fc_d,prioritized,mem_size, batch_size, replace, env_size = read_hp_json(load_path, file_name)
                learning_rate.append(lr)
                discount_rate.append(dr)
                epsilon.append(er)
                eps_min.append(er)
                eps_dec.append(er)
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
        for pr_i in positive_rewards:
            for nr_i in negative_rewards:
                for per_i in positive_exploration_rewards:
                    for nsr_i in negative_step_rewards:
                        for ms_i in max_steps:
                            for lr_i in learning_rate:
                                for dr_i in discount_rate:
                                    for er_i in epsilon:

                                        load_checkpoint = dqn(nr, training_sessions, episodes, dr_i, lr_i, er_i[0], er_i[1], er_i[2],
                                                    pr_i, nr_i, per_i, nsr_i, ms_i, i_exp,
                                                    n_actions, starting_beta, input_dims,
                                                    c_dims, k_size, s_size, fc_dims,
                                                    batch_size, mem_size, replace,
                                                    prioritized, models_path, load_path, save_path, load_checkpoint_path, env_size, load_checkpoint, start_up_exp)
                                        i_exp += 1
    else:
        for policy in policy_num:
            debug = True
            print("Testing policy %d:" %(policy))
            file_name = "hyperparameters%s.json" %(str(policy))
            ts, nr, ep, lr, dr, er, pr, ner, per, nsr, ms, n_actions, c_dims, k_size, s_size, fc_dims, prioritized, mem_size, batch_size, r, _ = read_hp_json(load_path, file_name)
            # int(ts),float(lr), float(dr), float(er), float(pr), float(nr), float(per), float(nsr), int(ms), int(n_actions), c_dims, k_size, s_size, fc_dims,int(mem_size), int(batch_size), int(replace),env_size
            
            agents = []
            for i in range(nr):
                agents.append(DQNAgent(nr, dr, er[0], er[1], er[2], lr,
                            n_actions, starting_beta, input_dims,
                            c_dims, k_size, s_size, fc_dims,
                            mem_size,
                            batch_size, r, prioritized,
                            algo='DQNAgent', env_name=env_size, chkpt_dir=models_path))
            for r_i, agent in enumerate(agents):
                file_name = "agent%s_experience%s_checkpoint.pth" %(str(r_i),str(policy))
                file_name = os.path.join(load_path, file_name)
                agent.q_eval.load_state_dict(T.load(file_name, map_location='cuda:0'))
                agent.q_eval.eval()
                agent.q_next.load_state_dict(T.load(file_name, map_location='cuda:0'))
                agent.q_next.eval()
            
            env = Environment(nr, pr, ner, per, nsr)

            PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])

            trajs = []
            steps = []
            cnt = [0]*nr
            trajectories = []
            fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))
            for i in range(0, testing_iterations):
                if i % 1000 == 0 and i != 0:
                    print("%d: %.2f %%, %.2f steps" %(int(i), float(cnt[0]+cnt[1])/float(i)*100, np.mean(np.array(steps))))
                observation = env.reset()

                trajectory = []

                done = False
                

                # if show_plot:
                #     PR.print_trajectories(ax, save_path, policy, env)
            
                for step in range(int(ms)):
                    if step == 0: actions = [None]
                    if show_plot:
                        plt.cla()
                        PR.print_trajectories(ax, save_path, policy, env, actions[0])
                        # breakpoint
                    actions = []
                    action = [0]*nr
                    for i_r in range(0,nr):
                        action[i_r] = agent.choose_action(observation[i_r])
                        trajectory.append((env.pos[i_r], action[i_r], i_r))
                    actions.append(action)
                    observation_, reward, done, info = env.step(action)
                    if info != None:
                        for j in range(nr):
                            if info == j:
                                cnt[j] += 1

                    observation = observation_

                    if done:
                        if show_plot:
                            plt.cla()
                            PR.print_trajectories(ax, save_path, policy, env, actions[0])
                            if not info: plt.pause(1)
                        trajectories.append(trajectory)
                        break
                    
                steps.append(step)
                # if step == int(ms)-1 and not done:
                #     trajectories.append(trajectory)

            p = (cnt[0]+cnt[1])/(testing_iterations)*100
            for i in range(nr):
                print("Percentage success for drone %d: %d / %d x 100 = %.2f %%" %(i, cnt[i], testing_iterations, (cnt[i])/(testing_iterations)*100))
            print("Total percentage success: %d / %d x 100 = %.2f %%" %(cnt[0]+cnt[1], testing_iterations, p))
            print("Average steps: %.2f" %(np.mean(np.array(steps))))
            file_name = "Results%s.json" %(str(policy))
            file_name = os.path.join(load_path, file_name)
            write_json("Success:%s, Average steps:%s" %(str(p), str(np.mean(np.array(steps)))), file_name)
breakpoint
