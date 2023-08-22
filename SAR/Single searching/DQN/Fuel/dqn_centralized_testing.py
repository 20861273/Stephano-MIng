import numpy as np
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from drqn_agent import DRQNAgent
from dqn_utils import read_json, write_json
from dqn_environment import Environment, HEIGHT, WIDTH, Point, States
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt

def test_centralized_dqn(policy_num, load_path, save_path, models_path, testing_iterations, show_plot, save_plot, test):

    for policy in policy_num:
        debug = True
        print("Testing policy %d:" %(policy))
        file_name = "hyperparameters.json"
        file_name = os.path.join(load_path, file_name)
        hp = read_json(file_name)

        # set input dimensions to encoding type
        if hp["encoding"] == "position":
            hp["input dims"] = [HEIGHT*WIDTH]
        elif hp["encoding"] == "position_exploration":
            hp["input dims"] = [HEIGHT*WIDTH*2]
        elif hp["encoding"] == "position_occupancy":
            hp["input dims"] = [HEIGHT*WIDTH*2]

        if hp["lidar"] and "image" not in hp["encoding"]:
            hp["input dims"][0] += 4

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
        if hp["agent type"] == "DQN":
            agent = DRQNAgent(hp["encoding"], hp["number of drones"], hp["discount rate"][0], hp["epsilon"][0][0], hp["epsilon"][0][1], hp["epsilon"][0][2], hp["learning rate"][0],
                            hp["n actions"], hp["starting beta"], hp["input dims"], hp["lidar"],
                            hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                            hp["mem size"], hp["batch size"], hp["replace"], hp["prioritized"],
                            algo='DQNAgent_distributed', env_name=model_name, chkpt_dir=load_models_path)
        elif hp["agent type"] == "DDQN":
            agent = DDQNAgent(hp["encoding"], hp["number of drones"], hp["discount rate"][0], hp["epsilon"][0][0], hp["epsilon"][0][1], hp["epsilon"][0][2], hp["learning rate"][0],
                            hp["n actions"], hp["starting beta"], hp["input dims"], hp["lidar"],
                            hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                            hp["mem size"], hp["batch size"], hp["replace"], hp["prioritized"],
                            algo='DDQNAgent_distributed', env_name=model_name, chkpt_dir=load_models_path)

        
        hp["curriculum learning"] = {"sparse reward": False, "collisions": False}
        checkpoint = agent.load_models()
        with T.no_grad():
            agent.q_eval.eval()
            agent.q_next.eval()
                            #nr, reward_system, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, training_type, encoding, lidar, curriculum_learning, episodes)
            env = Environment(hp["number of drones"], hp["obstacles"], hp["obstacle density"], hp["reward system"], hp["positive rewards"][0], hp["negative rewards"][0], hp["positive exploration rewards"][0], hp["negative step rewards"][0], hp["training type"], hp["encoding"], hp["lidar"], hp["curriculum learning"], 50000)

            PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])

            fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))

            i = 0
            if show_plot:
                    PR.print_trajectories(ax, save_path, policy, env)
                    if save_plot:
                        file_name = "p%dtrajectory%d%d.png" %(policy, i, 0)
                        plt.savefig(os.path.join(save_path, file_name))

            trajs = []
            steps = []
            cnt = 0
            trajectories = []
            timeout_cntr = 0
            collisions_grid = np.zeros(env.grid.shape)
            paths = []
            path = []
            starting_positions = []
            traj_actions = []
            obstacles = []
            
            if test == "grid":
                for x in range(WIDTH):
                    for y in range(HEIGHT):
                        if x == 0 or x == WIDTH-1:
                            continue
                        if y == 0 or y == HEIGHT-1:
                            continue
                        # if i % 100 == 0 and i != 0:
                        #     print("%d: %.2f %%, %.2f steps" %(int(i), float(cnt)/float(i)*100, np.mean(np.array(steps))))
                        image_observation, non_image_observation = env.reset(10000, 99)

                        env.exploration_grid[env.starting_pos[0].y, env.starting_pos[0].x] = False

                        obstacles.append(env.exploration_grid.copy())

                        env.grid[env.pos[0].y, env.pos[0].x] = States.UNEXP.value
                        env.exploration_grid[env.pos[0].y, env.pos[0].x] = False

                        env.starting_pos[0] = Point(x,y)
                        env.pos[0] = Point(x,y)
                        
                        env.grid[env.starting_pos[0].y, env.starting_pos[0].x] = States.ROBOT.value

                        trajectory = []
                        t_actions = []

                        done = False

                        starting_positions.append(env.starting_pos[0])
                    
                        for step in range(int(hp["max steps"][0])):
                            for i_r in range(0,hp["number of drones"]):
                                path.append(env.pos[i_r])
                            if len(path) > 120:
                                breakpoint
                            # if i == 24:
                            #     breakpoint
                            actions = []
                            action = [0]*hp["number of drones"]
                            for i_r in range(0,hp["number of drones"]):
                                if step == 0: action[i_r] = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"])
                                else: action[i_r] = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], previous_action[i_r])
                                previous_action = action.copy()
                            actions.append(action)
                            # trajectory.append((env.pos[0], action))
                            image_observation_, non_image_observation_, reward, done, info = env.step_centralized(action)
                            cnt += info[0]
                            for i_r in range(0,hp["number of drones"]):
                                t_actions.append(action[i_r])

                            image_observation = image_observation_
                            non_image_observation = non_image_observation_

                            collision_state = any(any(collision_tpye) for collision_tpye in info[1].values())
                            if collision_state:
                                for collision_type, collision_states in info[1].items():
                                    for i_r, collision_state in enumerate(collision_states):
                                        if collision_state:
                                            collisions_grid[env.pos[i_r].y, env.pos[i_r].x] += 1

                            elif info[0] == 0 and step == int(hp["max steps"][0])-1:
                                timeout_cntr += 1
                                path.append(info[0])
                                paths.append(path)
                                traj_actions.append(t_actions)
                                path = []
                                t_actions = []

                            if done:
                                path.append(info[0])
                                paths.append(path)
                                traj_actions.append(t_actions)
                                path = []
                                t_actions = []
                                if show_plot:
                                    plt.cla()
                                    PR.print_trajectories(ax, save_path, policy, env, actions[0], reward, done)
                                    if save_path:
                                        file_name = "p%dtrajectory%d%d.png" %(policy, i, step)
                                        plt.savefig(os.path.join(save_path, file_name))
                                trajectories.append(trajectory)
                                break
                            if show_plot:
                                plt.cla()
                                PR.print_trajectories(ax, save_path, policy, env, actions[0], reward)
                                if save_plot:
                                    file_name = "p%dtrajectory%d%d.png" %(policy, i, step)
                                    plt.savefig(os.path.join(save_path, file_name))
                        steps.append(step)
                        i += 1
                        # if step == int(ms)-1 and not done:
                        #     trajectories.append(trajectory)
                    path = []
                    t_actions = []

                if save_plot:
                    for cntr, path in enumerate(paths):
                        if path[-1] == 0:
                            del path[-1]
                            PR.print_graph(False, policy, path, traj_actions[cntr], starting_positions[cntr], obstacles[cntr], load_path, cntr, env)
                        # else:
                        #     del path[-1]
                        #     PR.print_graph(True, policy, path, traj_actions[cntr], starting_positions[cntr], obstacles[cntr], load_path, cntr, env)

                # p = cnt/(testing_iterations)*100
                p = cnt/((WIDTH-2)*(HEIGHT-2))*100
                print("Percentage success: %d / %d x 100 = %.2f %%" %(cnt, testing_iterations, p))
                print("Average steps: %.2f" %(np.mean(np.array(steps))))

                # file_name = "Results%s.json" %(str(policy))
                # file_name = os.path.join(load_path, file_name)
                # write_json("Success:%s, Average steps:%s, Average collisions:%s, Average timeouts:%s" %(str(p), str(np.mean(np.array(steps))), str(np.mean(collisions_grid)), str(timeout_cntr/testing_iterations)), file_name)


                print(collisions_grid)
                print("Average collisions: ", np.mean(collisions_grid))

                print("Timed out: ", timeout_cntr)
            elif test == "iterative":

            
                for i in range(0, testing_iterations):

                    if i % 100 == 0 and i != 0:
                        print("%d: %.2f %%, %.2f steps" %(int(i), float(cnt)/float(i)*100, np.mean(np.array(steps))))
                    image_observation, non_image_observation = env.reset(10000, 99)

                    env.exploration_grid[env.starting_pos[0].y, env.starting_pos[0].x] = False

                    obstacles.append(env.exploration_grid.copy())

                    env.exploration_grid[env.starting_pos[0].y, env.starting_pos[0].x] = True

                    trajectory = []
                    t_actions = []

                    done = False

                    starting_positions.append(env.starting_pos[0])

                    for step in range(int(hp["max steps"][0])):
                        for i_r in range(0,hp["number of drones"]):
                            path.append(env.pos[i_r])
                        actions = []
                        action = [0]*hp["number of drones"]
                        for i_r in range(0,hp["number of drones"]):
                            if step == 0: action[i_r] = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"])
                            else: action[i_r] = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], previous_action[i_r])
                            previous_action = action.copy()
                        actions.append(action)
                        # trajectory.append((env.pos[0], action))
                        image_observation_, non_image_observation_, reward, done, info = env.step_centralized(action)
                        cnt += info[0]
                        for i_r in range(0,hp["number of drones"]):
                            t_actions.append(action[i_r])

                        image_observation = image_observation_
                        non_image_observation = non_image_observation_

                        collision_state = any(any(collision_tpye) for collision_tpye in info[1].values())
                        if collision_state:
                            for collision_type, collision_states in info[1].items():
                                for i_r, collision_state in enumerate(collision_states):
                                    if collision_state:
                                        collisions_grid[env.pos[i_r].y, env.pos[i_r].x] += 1

                        elif info[0] == 0 and step == int(hp["max steps"][0])-1:
                            timeout_cntr += 1
                            path.append(info[0])
                            paths.append(path)
                            traj_actions.append(t_actions)
                            path = []
                            t_actions = []

                        if done:
                            path.append(info[0])
                            paths.append(path)
                            traj_actions.append(t_actions)
                            path = []
                            t_actions = []
                            if show_plot:
                                plt.cla()
                                PR.print_trajectories(ax, save_path, policy, env, actions[0], reward, done)
                                if save_path:
                                    file_name = "p%dtrajectory%d%d.png" %(policy, i, step)
                                    plt.savefig(os.path.join(save_path, file_name))
                            trajectories.append(trajectory)
                            break
                        if show_plot:
                            plt.cla()
                            PR.print_trajectories(ax, save_path, policy, env, actions[0], reward)
                            if save_plot:
                                file_name = "p%dtrajectory%d%d.png" %(policy, i, step)
                                plt.savefig(os.path.join(save_path, file_name))
                    steps.append(step)
                    i += 1
                    # if step == int(ms)-1 and not done:
                    #     trajectories.append(trajectory)

                if save_plot:
                    for cntr, path in enumerate(paths):
                        if path[-1] == 0:
                            del path[-1]
                            PR.print_graph(False, policy, path, traj_actions[cntr], starting_positions[cntr], obstacles[cntr], load_path, cntr, env)
                        else:
                            del path[-1]
                            PR.print_graph(True, policy, path, traj_actions[cntr], starting_positions[cntr], obstacles[cntr], load_path, cntr, env)

                p = cnt/(testing_iterations)*100
                print("Percentage success: %d / %d x 100 = %.2f %%" %(cnt, testing_iterations, p))
                print("Average steps: %.2f" %(np.mean(np.array(steps))))

                file_name = "Results%s.json" %(str(policy))
                file_name = os.path.join(load_path, file_name)
                write_json("Success:%s, Average steps:%s, Average collisions:%s, Average timeouts:%s" %(str(p), str(np.mean(np.array(steps))), str(np.mean(collisions_grid)), str(timeout_cntr/testing_iterations)), file_name)


                print(collisions_grid)
                print("Average collisions: ", np.mean(collisions_grid))

                print("Timed out: ", timeout_cntr)
