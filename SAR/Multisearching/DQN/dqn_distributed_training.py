import numpy as np
from dqn_agent import DQNAgent
from utils_decentralized import plot_learning_curve, write_json, read_hp_json, read_json, save_hp, read_checkpoint_hp_json
from dqn_environment import Environment, HEIGHT, WIDTH, Point, States, Direction
from datetime import datetime
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt
import time

# implement loss tracking

def distributed_dqn(nr, training_sessions, episodes, print_interval,
                    discount_rate, learning_rate, epsilon, eps_min, eps_dec,
                    positive_reward, negative_reward, positive_exploration_reward, negative_step_reward,
                    max_steps, i_exp,
                    n_actions, starting_beta, input_dims, 
                    c_dims, k_size, s_size, fc_dims,
                    batch_size, mem_size, replace,
                    prioritized, models_path, save_path, load_checkpoint_path, env_size, load_checkpoint):
    
    # initialize training session variables
    start_time = time.time()
    ts_rewards = []
    show_plot = False

    # training sessions loop
    for i_ts in range(training_sessions):
        # initialized in epoch variables
        rewards, steps, agents = [], [], []
        cntr = [0]*nr
        loss = [0]*nr

        # initialize environment
        env = Environment(nr, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward)
        
        # for debugging
        if show_plot:
            fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))
            PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])
        
        # initialize agents
        for i in range(nr):
            model_name = env_size + "_" + str(i)
            agents.append(DQNAgent(nr, discount_rate, epsilon, eps_min, eps_dec, learning_rate,
                        n_actions, starting_beta, input_dims,
                        c_dims, k_size, s_size, fc_dims,
                        mem_size, batch_size, replace, prioritized,
                        algo='DQNAgent_distributed', env_name=model_name, chkpt_dir=models_path))
        
        # load agent experiences and rewards
        if load_checkpoint:
            for r_i, agent in enumerate(agents):
                checkpoint = agent.load_models()
            i_ts = checkpoint['epoch']

            file_name = "ts_rewards%s.json" %(str(checkpoint['epoch']))
            file_name = os.path.join(load_checkpoint_path, file_name)
            ts_rewards = read_json(file_name)

            file_name = "rewards%s.json" %(str(checkpoint['epoch']))
            file_name = os.path.join(load_checkpoint_path, file_name)
            rewards = read_json(file_name)

        print("Experience: %s, Epoch: %s,\n\
            Discount rate: %s, Learning rate: %s, Epsilon: %s\n\
            Positive goal reward: %s, Negative collision reward:%s\n\
            Positive exploration reward: %s, Negative step reward:%s\n\
            Max steps: %s, Replace: %s"\
                 %(str(i_exp), str(i_ts), \
                    str(discount_rate), str(learning_rate), str(epsilon),\
                    str(positive_reward), str(negative_reward), str(positive_exploration_reward), str(negative_step_reward),\
                    str(max_steps), str(replace)))

        # epochs loop (multiple episodes are called an epoch)
        for i_episode in range(episodes):

            # if epoch loaded the epoch should start at the right episode
            if load_checkpoint:
                if i_episode <= checkpoint['episode']:
                    continue
                else:
                    load_checkpoint = False
            
            # initialize episodic variables
            episode_reward = [0]*nr
            action = [0]*nr

            # reset environment
            observation = env.reset()

            # episode loop
            for step in range(max_steps):

                # action selection
                for i_r, agent in enumerate(agents):
                    action[i_r] = agent.choose_action(observation[i_r])

                # execute step
                observation_, reward, done, goal_found_by = env.step_decentralized(action)

                # add step reward to episode reward
                temp_reward = [agnet_ep_reward + agent_reward for agnet_ep_reward, agent_reward in zip(episode_reward, reward)]
                episode_reward = temp_reward
                
                # update success counter
                if goal_found_by != None:
                    for j in range(nr):
                        if goal_found_by == j:
                            cntr[j] += 1

                # store transitions and execute learn function
                if not load_checkpoint:
                    for i_r, agent in enumerate(agents):
                        agent.store_transition(observation[i_r], action[i_r],
                                            reward[i_r], observation_[i_r], done)
                        loss[i_r] = agent.learn()

                observation = observation_

                # for debugging
                if show_plot:
                    plt.cla()
                    PR.print_trajectories(ax, save_path, i_ts, env, action)

                # checks if termination condition was met
                if done:
                    break
            
            # add episode rewards/steps to rewards/steps lists
            rewards.append(episode_reward)
            steps.append(step)

            # calculate average rewards over last 100 episodes (only for display purposes)
            avg_reward = []
            for i in range(nr):
                temp_avg_reward = [sum(agent_rewards[-100:][i] for agent_rewards in rewards) / len(rewards)]
                avg_reward = avg_reward + temp_avg_reward
            avg_steps = np.mean(steps[-100:])

            # save checkpoint
            if i_episode % 1000 == 0 and i_episode != 0:
                if not load_checkpoint:
                    print('... saving checkpoint ...')
                    for i_r, agent in enumerate(agents):
                        agent.save_models(i_ts, i_episode, time.time()-start_time, loss)

                        file_name = "ts_rewards%s.json" %(str(i_exp))
                        file_name = os.path.join(save_path, file_name)
                        write_json(ts_rewards, file_name)

                        file_name = "rewards%s.json" %(str(i_exp))
                        file_name = os.path.join(save_path, file_name)
                        write_json(rewards, file_name)

            # display progress
            if i_episode % print_interval==0 or i_episode == episodes-1 and i_episode != 0:
                for i in range(nr):
                    print('agent=', i,
                        'episode= ', i_episode,
                            ',reward= %.2f,' % episode_reward[i],
                            'average_reward= %.2f,' % avg_reward[i],
                            'average_steps= %.2f,' % avg_steps,
                            'success= %.4f' % (float(cntr[i])/float(print_interval)*100.0))
                print('total success= %.4f' % (float(sum(cntr))/float(print_interval)*100.0))
                print(agents[0].epsilon)
                
                cntr = [0]*nr
                
        # save rewards in training session rewards
        ts_rewards.append(rewards)

        # save model and rewards
        if not load_checkpoint:
            for i_r, agent in enumerate(agents):
                agent.save_models(i_ts, i_episode, time.time()-start_time, loss)

                file_name = "ts_rewards%s.json" %(str(i_exp))
                file_name = os.path.join(save_path, file_name)
                write_json(ts_rewards, file_name)

                file_name = "rewards%s.json" %(str(i_exp))
                file_name = os.path.join(save_path, file_name)
                write_json(rewards, file_name)
    
    # if loaded from wrong training session and all epochs have been trained to final episode,
    # this makes sure the load_checkpoint variable is false for the next training session
    if load_checkpoint:
        load_checkpoint = False

    # calculate total time of training session
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)

    # save reward lists and plot learning curve
    # explanation of reward data being configured for saving
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
    
    file_name = "rewards%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(ts_rewards, file_name)

    return load_checkpoint