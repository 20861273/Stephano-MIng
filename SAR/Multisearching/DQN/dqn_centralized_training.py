from dqn_environment import Environment, HEIGHT, WIDTH, Point, States, Direction
from dqn_utils import plot_learning_curve, write_json, read_json
from dqn_agent import DQNAgent
from dqn_save_results import print_results
import numpy as np
import os
import torch as T
import time
import matplotlib.pyplot as plt

def centralized_dqn(nr, training_sessions, episodes, print_interval, training_type, encoding,
                    curriculum_learning,
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
        rewards, steps = [], []
        cntr = 0

        # initialize environment
        env = Environment(nr, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, training_type, encoding, curriculum_learning, episodes)
        
        # initialize agents
        model_name = str(i_exp) + "_" + env_size
        agent = DQNAgent(nr, discount_rate, epsilon, eps_min, eps_dec, learning_rate,
                        n_actions, starting_beta, input_dims,
                        c_dims, k_size, s_size, fc_dims,
                        mem_size, batch_size, replace, prioritized,
                        algo='DQNAgent_distributed', env_name=model_name, chkpt_dir=models_path)
        
        # for debugging
        fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))
        PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])
        
        # load agent experiences and rewards
        if load_checkpoint:
            checkpoint = agent.load_models() 

            if i_ts < checkpoint['epoch']:
                continue           

            file_name = "ts_rewards%s.json" %(str(checkpoint['session']))
            file_name = os.path.join(load_checkpoint_path, file_name)
            ts_rewards = read_json(file_name)

            file_name = "rewards%s.json" %(str(checkpoint['session']))
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
            # if i_episode > 101 and i_ts == 1:
            #     quit()
            # if i_episode == 201:
            #     breakpoint

            # if epoch loaded the epoch should start at the right episode
            if load_checkpoint:
                if i_episode <= checkpoint['episode']:
                    continue
                else:
                    load_checkpoint = False
            
            # initialize episodic variables
            episode_reward = 0
            action = [0]*nr
            loss = 0

            # reset environment
            observation = env.reset(i_episode)
            
            # episode loop
            for step in range(max_steps):

                # action selection
                for i_r in range(0,nr):
                    action[i_r] = agent.choose_action(observation[i_r])
                
                # execute step
                observation_, reward, done, info = env.step_centralized(action)
                
                # add step reward to episode reward
                episode_reward += reward
                
                # update success counter
                cntr += info

                # store transitions and execute learn function
                if not load_checkpoint:
                    for i_r in range(0,nr):
                        agent.store_transition(observation[i_r], action[i_r],
                                            reward, observation_[i_r], done)
                        loss = agent.learn()

                observation = observation_

                # for debugging
                # Done not tested
                #################################################################################
                if show_plot:
                    plt.cla()
                    PR.print_trajectories(ax, save_path, i_ts, env, action)
                #################################################################################

                # checks if termination condition was met
                if done:
                    break

            # add episode rewards/steps to rewards/steps lists
            rewards.append(episode_reward)
            steps.append(step)

            # calculate average rewards over last 100 episodes (only for display purposes)
            avg_reward = np.mean(rewards[-100:])
            avg_steps = np.mean(steps[-100:])

            # save checkpoint
            if i_episode % 10000 == 0 and i_episode != 0:
                if not load_checkpoint:
                    print('... saving checkpoint ...')
                    agent.save_models(i_exp, i_ts, i_episode, time.time()-start_time, loss)

                    file_name = "ts_rewards%s.json" %(str(i_exp))
                    file_name = os.path.join(save_path, file_name)
                    write_json(ts_rewards, file_name)

                    file_name = "rewards%s.json" %(str(i_exp))
                    file_name = os.path.join(save_path, file_name)
                    write_json(rewards, file_name)

            # display progress
            if i_episode % print_interval == 0 or i_episode == episodes-1 and i_episode != 0:
                print('episode= ', i_episode,
                        ',reward= %.2f,' % episode_reward,
                        'average_reward= %.2f,' % avg_reward,
                        'average_steps= %.2f,' % avg_steps,
                        'loss=%.2f' % loss,
                        'success= %.4f' % (float(cntr)/1000.0*100.0))
                cntr = 0
            
        # save rewards in training session rewards
        ts_rewards.append(rewards)

        # save model and rewards
        if not load_checkpoint:
            agent.save_models(i_exp, i_ts, i_episode, time.time()-start_time, loss)

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
    filename = 'learning_cruve%s.png' %(str(i_exp))
    filename = os.path.join(save_path, filename)
    plot_learning_curve(nr, training_type, ts_rewards, filename, \
        learning_rate, discount_rate, epsilon, \
            positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, \
                max_steps, total_time)
    
    file_name = "ts_rewards%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(ts_rewards, file_name)

    file_name = "rewards%s.json" %(str(i_exp))
    file_name = os.path.join(save_path, file_name)
    write_json(rewards, file_name)

    return load_checkpoint