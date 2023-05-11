# Notes:
# Last at line 

import numpy as np
from dqn_agent_simultaneous import DQNAgent
from utils import plot_learning_curve, write_json, read_hp_json, read_json, save_hp, read_checkpoint_hp_json
from dqn_environment_simultaneous import Environment, HEIGHT, WIDTH, Point, States, Direction
from datetime import datetime
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt
import time
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def dqn(nr, training_sessions, episodes, discount_rate, learning_rate, epsilon, eps_min, eps_dec,
        positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, max_steps, i_exp,
        n_actions, input_dims, 
        c_dims, k_size, s_size, fc_dims,
        batch_size, mem_size, replace,
        models_path, env_size, load_checkpoint, start_up_exp):
    start_time = time.time()
    ts_rewards = []
    for i_ts in range(training_sessions):
        rewards, steps = [], []
        env = Environment(nr, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward)
        agent = DQNAgent(nr, discount_rate, epsilon, eps_min, eps_dec, learning_rate,
                     n_actions, input_dims,
                     c_dims, k_size, s_size, fc_dims,
                     mem_size=mem_size,
                     batch_size=batch_size, replace=replace,
                     algo='DQNAgent', env_name=env_size, chkpt_dir=models_path)
        if load_checkpoint:
            file_name = "experience%s_checkpoint.pth" %(str(start_up_exp))
            file_name = os.path.join(load_path, file_name)
            agent.q_eval.load_state_dict(T.load(file_name))
            agent.q_eval.eval()
            agent.q_next.load_state_dict(T.load(file_name))
            agent.q_next.eval()

            file_name = "hyperparameters%s_checkpoint.json" %(str(start_up_exp))
            file_name = os.path.join(load_path, file_name)
            ts, ep = read_checkpoint_hp_json(file_name)

            file_name = "ts_rewards%s.json" %(str(start_up_exp))
            file_name = os.path.join(load_path, file_name)
            ts_rewards = read_json(file_name)

            file_name = "rewards%s.json" %(str(start_up_exp))
            file_name = os.path.join(load_path, file_name)
            rewards = read_json(file_name)

        if load_checkpoint:
            if i_ts < ts:
                continue

        print("Experience: %s, Epoch: %s,\n\
            Discount rate: %s, Learning rate: %s, Epsilon: %s\n\
            Positive goal reward: %s, Negative collision reward:%s\n\
            Positive exploration reward: %s, Negative step reward:%s\n\
            Max steps: %s, Replace: %s"\
                 %(str(i_exp), str(i_ts), \
                    str(discount_rate), str(learning_rate), str(epsilon),\
                    str(positive_reward), str(negative_reward), str(positive_exploration_reward), str(negative_step_reward),\
                    str(max_steps), str(replace)))

        cntr = 0
        best_reward = -np.inf
        for i_episode in range(episodes):
            if load_checkpoint:
                if i_episode <= ep:
                    continue
                else:
                    load_checkpoint = False
            episode_reward = 0
            done = False
            observation = env.reset()
            breakpoint
            action = [0]*nr
            for step in range(max_steps):
                for i_r in range(0,nr):
                    action[i_r] = agent.choose_action(observation[i_r])
                observation_, reward, done, info = env.step(action)
                episode_reward += reward
                cntr += info

                if not load_checkpoint:
                    for i_r in range(0,nr):
                        agent.store_transition(observation[i_r], action[i_r],
                                            reward, observation_[i_r], done)
                        agent.learn()

                observation = observation_

                if done:
                    break
            rewards.append(episode_reward)
            steps.append(step)

            avg_reward = np.mean(rewards[-100:])
            avg_steps = np.mean(steps[-100:])
            if i_episode % 50 == 0 and i_episode != 0:
                if not load_checkpoint:
                    file_name = "experience%s_checkpoint.pth" %(str(i_exp))
                    file_name = os.path.join(save_path, file_name)
                    T.save(agent.q_eval.state_dict(),file_name)
                    
                    hp =    {
                            "current_training_session":i_ts,
                            "current_episode":i_episode
                            }
                    file_name = "hyperparameters%s_checkpoint.json" %(str(i_exp))
                    file_name = os.path.join(save_path, file_name)
                    write_json(hp, file_name)

                    file_name = "ts_rewards%s.json" %(str(i_exp))
                    file_name = os.path.join(save_path, file_name)
                    write_json(ts_rewards, file_name)

                    file_name = "rewards%s.json" %(str(i_exp))
                    file_name = os.path.join(save_path, file_name)
                    write_json(rewards, file_name)

            if i_episode % 50==0 or i_episode == episodes-1 and i_episode != 0:
                print('episode= ', i_episode,
                        ',reward= %.2f,' % episode_reward,
                        'average_reward= %.2f,' % avg_reward,
                        'average_steps= %.2f,' % avg_steps,
                        'success= %.4f' % (float(cntr)/50.0*100.0))
                cntr = 0
            
        ts_rewards.append(rewards)
        if not load_checkpoint:
            file_name = "experience%s.pth" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            T.save(agent.q_eval.state_dict(),file_name)

            file_name = "experience%s_checkpoint.pth" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            T.save(agent.q_eval.state_dict(),file_name)

            hp =    {
                    "current_training_session":i_ts,
                    "current_episode":i_episode
                    }
            file_name = "hyperparameters%s_checkpoint.json" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            write_json(hp, file_name)

            file_name = "ts_rewards%s.json" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            write_json(ts_rewards, file_name)

            file_name = "rewards%s.json" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            write_json(rewards, file_name)
    
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
    hp =    {
            "training_session":training_sessions,
            "learning_rate":learning_rate,
            "discount_rate":discount_rate,
            "epsilon":epsilon,
            "positive_goal":positive_reward,
            "negative_collision":negative_reward,
            "negative_step":positive_exploration_reward,
            "positive_exploration":negative_step_reward,
            "max_steps":max_steps,
            "n_actions":n_actions,
            "c_dims":c_dims,
            "k_size":k_size,
            "s_size":s_size,
            "fc_dims":fc_dims,
            "mem_size":mem_size,
            "batch_size":batch_size,
            "replace":replace,
            "env_size":env_size
            }
    file_name = "hyperparameters%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(hp, file_name)

if __name__ == '__main__':
    # Testing: for on-policy runs
    off_policy = True
    show_plot = False
    policy_num = [0]
    testing_iterations = 10000

    load_checkpoint = True
    n_experiences = 4
    start_up_exp = 1

    nr = 2

    # states:
    # observation states: "position", "position_explored", "image"
    # termination states: "goal", "covered"
    observation_state = "image"
    termination_state = "goal"

    # initialize hyperparameters
    learning_rate = [0.001,0.0001]
    discount_rate = [0.5,0.9]
    epsilon = [0.01]
    eps_min = [0.01]

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
        input_dims = (3,HEIGHT, WIDTH)

    training_sessions = 3
    episodes = 100
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
                file_name = "hyperparameters%s.json" %(str(start_up_exp))
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
        for pr_i in positive_rewards:
            for nr_i in negative_rewards:
                for per_i in positive_exploration_rewards:
                    for nsr_i in negative_step_rewards:
                        for ms_i in max_steps:
                            for lr_i in learning_rate:
                                for dr_i in discount_rate:
                                    for er_i in epsilon:
                                        if load_checkpoint and i_exp < start_up_exp:
                                            i_exp += 1
                                            continue
                                        rewards = dqn(nr, training_sessions, episodes, dr_i, lr_i, er_i, er_i, er_i,
                                                    pr_i, nr_i, per_i, nsr_i, ms_i, i_exp,
                                                    n_actions, input_dims,
                                                    c_dims, k_size, s_size, fc_dims,
                                                    batch_size, mem_size, replace,
                                                    models_path, env_size, load_checkpoint, start_up_exp)
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
