from dqn_environment import Environment, HEIGHT, WIDTH, Point, States, Direction
from utils import plot_learning_curve, write_json, read_hp_json, read_json, save_hp, read_checkpoint_hp_json
from dqn_agent import DQNAgent
import numpy as np
import os
import torch as T
import time

def test_dqn(nr, training_sessions, episodes, discount_rate, learning_rate, epsilon, eps_min, eps_dec,
        positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, max_steps, i_exp,
        n_actions, input_dims, 
        c_dims, k_size, s_size, fc_dims,
        batch_size, mem_size, replace,
        models_path, load_path, save_path, env_size, load_checkpoint, start_up_exp):
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
            if i_episode % 10000 == 0 and i_episode != 0:
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

            if i_episode % 1000==0 or i_episode == episodes-1 and i_episode != 0:
                print('episode= ', i_episode,
                        ',reward= %.2f,' % episode_reward,
                        'average_reward= %.2f,' % avg_reward,
                        'average_steps= %.2f,' % avg_steps,
                        'success= %.4f' % (float(cntr)/1000.0*100.0))
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
            "number_of_drones":nr,
            "episodes": episodes,
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