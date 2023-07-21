from dqn_environment import Environment, HEIGHT, WIDTH, Point, States, Direction
from dqn_utils import plot_learning_curve, write_json, read_json
from dqn_agent import DQNAgent
from drqn_agent import DRQNAgent
from ddqn_agent import DDQNAgent
from dqn_save_results import print_results
import numpy as np
import os
import torch as T
import time
import matplotlib.pyplot as plt

def centralized_dqn(nr, obstacles, obstacle_density, training_sessions, episodes, print_interval, training_type, agent_type, encoding,
                    curriculum_learning, reward_system, allow_windowed_revisiting,
                    discount_rate, learning_rate, epsilon, eps_min, eps_dec,
                    positive_reward, negative_reward, positive_exploration_reward, negative_step_reward,
                    max_steps, i_exp,
                    n_actions, starting_beta, input_dims, lidar,
                    c_dims, k_size, s_size, fc_dims,
                    batch_size, mem_size, replace,
                    prioritized, models_path, save_path, load_checkpoint_path, env_size, load_checkpoint, device_num):
    
    # initialize training session variables
    start_time = time.time()
    ts_rewards, ts_steps, ts_losses = [], [], []
    show_plot, skip = False, False

    # training sessions loop
    for i_ts in range(training_sessions):
        # initialized in epoch variables
        rewards, steps, episode_loss = [], [], []
        cntr, timeout_cntr = 0, 0
        percentage = 0.0

        # initialize environment
        env = Environment(nr, obstacles, obstacle_density, reward_system, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, training_type, encoding, lidar, curriculum_learning, episodes)
        
        collisions_grid = np.zeros(env.grid.shape)
        previous_avg_collisions = 0

        # initialize agents
        model_name = str(i_exp) + "_" + env_size
        if agent_type == "DQN":
            agent = DQNAgent(encoding, nr, discount_rate, epsilon, eps_min, eps_dec, learning_rate,
                            n_actions, starting_beta, input_dims, lidar,
                            c_dims, k_size, s_size, fc_dims,
                            mem_size, batch_size, replace, prioritized,
                            algo='DQNAgent_distributed', env_name=model_name, chkpt_dir=models_path, device_num=device_num)
        elif agent_type == "DDQN":
            agent = DDQNAgent(encoding, nr, discount_rate, epsilon, eps_min, eps_dec, learning_rate,
                            n_actions, starting_beta, input_dims, lidar,
                            c_dims, k_size, s_size, fc_dims,
                            mem_size, batch_size, replace, prioritized,
                            algo='DDQNAgent_distributed', env_name=model_name, chkpt_dir=models_path, device_num=device_num)
        
        # for debugging
        fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))
        PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])
        
        # load agent experiences and rewards
        if load_checkpoint:
            if int(os.listdir(models_path)[-1][0]) == i_exp:
                checkpoint = agent.load_models()
                if checkpoint["episode"] == episodes-1: skip = True

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
                    print('episode', i_episode)
            
            # initialize episodic variables
            episode_reward = 0
            average_loss = 0
            action = [0]*nr
            loss = 0

            # reset environment
            percentage = float(cntr)/float(print_interval)*100.0
            image_observation, non_image_observation = env.reset(i_episode, percentage)
            # for debugging
            if show_plot:
                plt.cla()
                PR.print_trajectories(ax, save_path, i_ts, env)
            
            # episode loop
            step = 0
            done = False
            while not done and step < max_steps:
                if len(non_image_observation) > 1:
                    breakpoint
                # action selection
                for i_r in range(0,nr):
                    if step == 0: action[i_r] = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], allow_windowed_revisiting)
                    else: action[i_r] = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], allow_windowed_revisiting, previous_action[i_r])
                    previous_action = action.copy()
                
                # execute step
                image_observation_, non_image_observation_, reward, done, info = env.step_centralized(action)
                
                # add step reward to episode reward
                episode_reward += reward
                
                # update success counter
                cntr += info[0]

                # store transitions and execute learn function
                if not load_checkpoint:
                    for i_r in range(0,nr):
                        agent.store_transition(image_observation[i_r], non_image_observation[i_r], action[i_r],
                                            reward, image_observation_[i_r], non_image_observation_[i_r], done)
                        loss = agent.learn()
                        if not loss == None: average_loss += float(loss)

                image_observation = image_observation_
                non_image_observation = non_image_observation_

                collision_state = any(any(collision_tpye) for collision_tpye in info[1].values())
                if collision_state: # collisions counter
                    for collision_type, collision_states in info[1].items():
                        for i_r, collision_state in enumerate(collision_states):
                            if collision_state:
                                collisions_grid[env.pos[i_r].y, env.pos[i_r].x] += 1
                elif info[0] == 0 and step == max_steps-1: # timeout counter
                    timeout_cntr += 1

                # for debugging
                if show_plot:
                    plt.cla()
                    PR.print_trajectories(ax, save_path, i_ts, env, action, reward, info[0])

                # checks if termination condition was met
                if done:
                    # for debugging
                    if show_plot:
                        plt.cla()
                        PR.print_trajectories(ax, save_path, i_ts, env, action, reward, info[0])
                    break
                step += 1

            # add episode rewards/steps to rewards/steps lists
            rewards.append(episode_reward)
            steps.append(step)
            episode_loss.append(tuple((i_episode, round(average_loss, 5))))

            # calculate average rewards over last 100 episodes (only for display purposes)
            avg_reward = np.mean(rewards[-100:])
            avg_steps = np.mean(steps[-100:])
            avg_collisions = np.mean(collisions_grid)

            # save checkpoint
            if i_episode % 1000 == 0 and i_episode != 0:
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
                percentage = float(cntr)/float(print_interval)*100.0
                if loss == None: loss = 0
                print('episode= ', i_episode,
                        ',reward= %.2f,' % episode_reward,
                        'average_reward= %.2f,' % avg_reward,
                        'average_steps= %.2f,' % avg_steps,
                        'collisions= %.2f,' % (avg_collisions-previous_avg_collisions),
                        'loss=%f' % loss,
                        'success= %.4f' % (percentage))
                cntr = 0
                previous_avg_collisions = avg_collisions
            
        # save rewards in training session rewards
        ts_rewards.append(rewards)
        ts_steps.append(steps)
        ts_losses.append(episode_loss)

        # save model and rewards
        if not load_checkpoint:
            agent.save_models(i_exp, i_ts, i_episode, time.time()-start_time, loss)

            file_name = "ts_rewards%s.json" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            write_json(ts_rewards, file_name)

            file_name = "rewards%s.json" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            write_json(rewards, file_name)

            file_name = "ts_steps%s.json" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            write_json(ts_steps, file_name)

            file_name = "ts_loss%s.json" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            write_json(ts_losses, file_name)

            file_name = "collisions%s.json" %(str(i_exp))
            file_name = os.path.join(save_path, file_name)
            write_json(collisions_grid.tolist(), file_name)
    
    # if loaded from wrong training session and all epochs have been trained to final episode,
    # this makes sure the load_checkpoint variable is false for the next training session
    if load_checkpoint:
        load_checkpoint = False

    if not skip:
    
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

        file_name = "ts_steps%s.json" %(str(i_exp))
        file_name = os.path.join(save_path, file_name)
        write_json(ts_steps, file_name)

        file_name = "ts_loss%s.json" %(str(i_exp))
        file_name = os.path.join(save_path, file_name)
        write_json(ts_losses, file_name)

        file_name = "collisions%s.json" %(str(i_exp))
        file_name = os.path.join(save_path, file_name)
        write_json(collisions_grid.tolist(), file_name)

    return load_checkpoint