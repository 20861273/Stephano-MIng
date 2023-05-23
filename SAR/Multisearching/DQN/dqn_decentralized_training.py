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

def dqn(nr, training_sessions, episodes, discount_rate, learning_rate, epsilon, eps_min, eps_dec,
        positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, max_steps, i_exp,
        n_actions, starting_beta, input_dims, 
        c_dims, k_size, s_size, fc_dims,
        batch_size, mem_size, replace,
        prioritized, models_path, load_path, save_path, env_size, load_checkpoint, start_up_exp):
    start_time = time.time()
    ts_rewards = []
    show_plot = False
    for i_ts in range(training_sessions):
        rewards, steps = [], []
        env = Environment(nr, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward)
        if show_plot:
            fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))
            PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])
        agents = []
        for i in range(nr):
            agents.append(DQNAgent(nr, discount_rate, epsilon, eps_min, eps_dec, learning_rate,
                        n_actions, starting_beta, input_dims,
                        c_dims, k_size, s_size, fc_dims,
                        mem_size, batch_size, replace, prioritized,
                        algo='DQNAgent', env_name=env_size, chkpt_dir=models_path))
        if load_checkpoint:
            for r_i, agent in enumerate(agents):
                file_name = "agent%s_experience%s_checkpoint.pth" %(str(r_i),str(start_up_exp))
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

        cntr = [0]*nr
        best_reward = -np.inf
        for i_episode in range(episodes):
            if load_checkpoint:
                if i_episode <= ep:
                    continue
                else:
                    load_checkpoint = False
            episode_reward = [0]*nr
            done = False
            observation = env.reset()
            breakpoint
            action = [0]*nr
            for step in range(max_steps):
                for i_r, agent in enumerate(agents):
                    action[i_r] = agent.choose_action(observation[i_r])
                observation_, reward, done, info = env.step(action)
                temp_reward = [agnet_ep_reward + agent_reward for agnet_ep_reward, agent_reward in zip(episode_reward, reward)]
                episode_reward = temp_reward
                # cntr += info
                if info != None:
                    for j in range(nr):
                        if info == j:
                            cntr[j] += 1

                if not load_checkpoint:
                    for i_r, agent in enumerate(agents):
                        agent.store_transition(observation[i_r], action[i_r],
                                            reward[i_r], observation_[i_r], done)
                        agent.learn()

                observation = observation_

                if show_plot:
                    plt.cla()
                    PR.print_trajectories(ax, save_path, i_ts, env, action)

                if done:
                    break
            rewards.append(episode_reward)
            steps.append(step)

            avg_reward = []
            for i in range(nr):
                temp_avg_reward = [sum(agent_rewards[-100:][i] for agent_rewards in rewards) / len(rewards)]
                avg_reward = avg_reward + temp_avg_reward
            avg_steps = np.mean(steps[-100:])
            if i_episode % 10000 == 0 and i_episode != 0:
                if not load_checkpoint:
                    for i_r, agent in enumerate(agents):
                        file_name = "agent%s_experience%s_checkpoint.pth" %(str(i_r), str(i_exp))
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

            if i_episode % 1000==0 or i_episode == episodes-1 and i_episode != 0:
                for i in range(nr):
                    print('agent=', i,
                        'episode= ', i_episode,
                            ',reward= %.2f,' % episode_reward[i],
                            'average_reward= %.2f,' % avg_reward[i],
                            'average_steps= %.2f,' % avg_steps,
                            'success= %.4f' % (float(cntr[i])/1000.0*100.0))
                print('total success= %.4f' % (float(sum(cntr))/1000.0*100.0))
                cntr = [0]*nr
                
            
        ts_rewards.append(rewards)
        if not load_checkpoint:
            for i_r, agent in enumerate(agents):
                file_name = "agent%s_experience%s_checkpoint.pth" %(str(i_r), str(i_exp))
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
    
    if load_checkpoint:
        load_checkpoint = False

    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)

    # ts_rewards        = [[[drone_0_episode_0_reward, drone_1_episode_0_reward,...,n_drones],
    #                       [drone_0_episode_1_reward, drone_1_episode_1_reward,...,n_drones],
    #                       ...,n_episodes],
    #                       ...,n_epochs]
    # temp_ts_rewards   = [[[drone_0_episode_0_reward,...,n_episdoes],
    #                       [drone_1_episode_0_reward,...,n_episdoes],
    #                       ...,n_drones],
    #                       ...,n_epochs]
    # ts_rewards        = [[epoch_0_drone_0_episode_0_reward,...,n_episdoes],
    #                      ...,
    #                      [epoch_m_drone_0_episode_0_reward,...,n_episdoes],
    #                      ...,
    #                      [epoch_m_drone_r_episode_0_reward,...,n_episdoes]] 
    temp_ts_rewards = [[list(sub) for sub in zip(*sublist)] for sublist in ts_rewards]
    ts_rewards = []
    for r_i in range(nr):
        filename = 'drone%s_learning_cruve%s.png' %(str(r_i), str(i_exp))
        filename = os.path.join(save_path, filename)
        for n_ts_rewards in temp_ts_rewards:
            ts_rewards.append(n_ts_rewards[r_i])
        plot_learning_curve(r_i, ts_rewards[r_i*training_sessions:r_i*training_sessions+training_sessions], filename, \
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
    return load_checkpoint