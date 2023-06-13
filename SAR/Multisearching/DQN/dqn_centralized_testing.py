import numpy as np
from dqn_agent import DQNAgent
from dqn_utils import read_json, write_json
from dqn_environment import Environment, HEIGHT, WIDTH
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt

def test_centralized_dqn(policy_num, load_path, save_path, models_path, testing_iterations, show_plot, save_plot):

    for policy in policy_num:
        debug = True
        print("Testing policy %d:" %(policy))
        file_name = "hyperparameters.json"
        file_name = os.path.join(load_path, file_name)
        hp = read_json(file_name)

        load_models_path = os.path.join(load_path, 'models')

        # fills hps but doesn't work yet
        hp_lens = {}
        for key in hp:
            if key == "n actions": break
            if type(hp[key]) == list: hp_lens[key] = len(hp[key])
        max_len = max(hp_lens.values())
        for key in hp_lens:
            if hp_lens[key] < max_len:
                [hp[key].append(hp[key][0]) for i in range(max_len-1)]

        # print("Discount rate: %s, Learning rate: %s, Epsilon: %s"\
        #          %(str(hp["discount rate"][policy]), str(hp["learning rate"][policy]), str(hp["epsilon"][policy])))

        model_name = str(policy) + "_" + hp["env size"]
        # model_name = hp["env size"]
        agent = DQNAgent(hp["number of drones"], hp["discount rate"][0], hp["epsilon"][0][0], hp["epsilon"][0][1], hp["epsilon"][0][2], hp["learning rate"][0],
                        hp["n actions"], hp["starting beta"], hp["input dims"],
                        hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                        hp["mem size"], hp["batch size"], hp["replace"], hp["prioritized"],
                        algo='DQNAgent_distributed', env_name=model_name, chkpt_dir=load_models_path)
        
        checkpoint = agent.load_models()
        agent.q_eval.eval()
        agent.q_next.eval()

        env = Environment(hp["number of drones"], hp["positive rewards"][0], hp["negative rewards"][0], hp["positive exploration rewards"][0], hp["negative step rewards"][0], hp["training type"], hp["encoding"], {"collisions": False, "sparse reward": False}, 50000)

        PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])

        trajs = []
        steps = []
        cnt = 0
        trajectories = []
        fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))
        for i in range(0, testing_iterations):
            if i % 100 == 0 and i != 0:
                print("%d: %.2f %%, %.2f steps" %(int(i), float(cnt)/float(i)*100, np.mean(np.array(steps))))
            observation = env.reset(0)

            trajectory = []

            done = False
            

            if show_plot:
                PR.print_trajectories(ax, save_path, policy, env)
                if save_plot:
                    file_name = "p%dtrajectory%d%d.png" %(policy, i, 0)
                    plt.savefig(os.path.join(save_path, file_name))
        
            for step in range(int(hp["max steps"][0])):
                actions = []
                action = [0]*hp["number of drones"]
                for i_r in range(0,hp["number of drones"]):
                    action[i_r] = agent.choose_action(observation[i_r])
                actions.append(action)
                trajectory.append((env.pos[i_r], action, i_r))
                observation_, reward, done, info = env.step_centralized(action)
                cnt += info

                observation = observation_

                if done:
                    if save_plot:
                        plt.cla()
                        PR.print_trajectories(ax, save_path, policy, env, actions[0])
                        if save_path:
                            file_name = "p%dtrajectory%d%d.png" %(policy, i, step)
                            plt.savefig(os.path.join(save_path, file_name))
                    trajectories.append(trajectory)
                    break
                if show_plot:
                    plt.cla()
                    PR.print_trajectories(ax, save_path, policy, env, actions[0])
                    if save_plot:
                        file_name = "p%dtrajectory%d%d.png" %(policy, i, step)
                        plt.savefig(os.path.join(save_path, file_name))
            steps.append(step)
            # if step == int(ms)-1 and not done:
            #     trajectories.append(trajectory)

        p = cnt/(testing_iterations)*100
        print("Percentage success: %d / %d x 100 = %.2f %%" %(cnt, testing_iterations, p))
        print("Average steps: %.2f" %(np.mean(np.array(steps))))

        file_name = "Results%s.json" %(str(policy))
        file_name = os.path.join(load_path, file_name)
        write_json("Success:%s, Average steps:%s" %(str(p), str(np.mean(np.array(steps)))), file_name)
