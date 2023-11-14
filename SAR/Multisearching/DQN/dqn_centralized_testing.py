import numpy as np
from dqn_agent import DQNAgent
from drqn_agent import DRQNAgent
from duel_dqn_agent import DuelDQNAgent
from duel_ddqn_agent import DuelDDQNAgent
from dqn_utils import read_json, write_json
# from dqn_environment import Environment, HEIGHT, WIDTH, States, Point, Direction
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt

def test_centralized_dqn(policy_num, session, load_path, save_path, models_path, testing_iterations, goal_spawning, show_plot, save_plot, test):

    for policy in policy_num:
        debug = True
        print("\nTesting policy %d:" %(policy))
        file_name = "hyperparameters.json"
        file_name = os.path.join(load_path, file_name)
        hp = read_json(file_name)

        if hp["n actions"] == 3:
            from dqn_new_agent import DQNAgent
            from dqn_new_environment import Environment, HEIGHT, WIDTH, Point, States, Direction
        else:
            from dqn_agent import DQNAgent
            from dqn_environment import Environment, HEIGHT, WIDTH, Point, States, Direction

        max_steps = HEIGHT*WIDTH*3
        
        for ts in range(hp["training sessions"]):
            if len(session) > 0:
                if ts not in session: continue
            print("Session: ", ts)
            # set input dimensions to encoding type
            if hp["encoding"] == "position":
                hp["input dims"] = [HEIGHT*WIDTH]
            elif hp["encoding"] == "position_exploration":
                hp["input dims"] = [HEIGHT*WIDTH*2]
            elif hp["encoding"] == "position_occupancy":
                hp["input dims"] = [HEIGHT*WIDTH*2]
            elif hp["encoding"] == "local":
                hp["input dims"] = (4)#+2*(hp["number of drones"]-1))

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

            # model_name = str(policy) + "_" + hp["env size"]
            model_name = str(policy) + "_" + str(ts) + "_" + hp["env size"]
            # model_name = hp["env size"]
            if hp["agent type"] == "DQN":
                agent = DQNAgent(hp["encoding"], hp["number of drones"], hp["discount rate"][0], 0, 0, 0, hp["learning rate"][0],
                                hp["n actions"], hp["starting beta"], hp["input dims"], hp["guide"], hp["lidar"], hp["lstm"],
                                hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                                hp["mem size"], hp["batch size"], hp["replace"], hp['nstep'], hp["nstep N"], hp["prioritized"],
                                algo='DQNAgent_centralized_turn', env_name=model_name, chkpt_dir=load_models_path)
            elif hp["agent type"] == "DDQN":
                agent = DQNAgent(hp["encoding"], hp["number of drones"], hp["discount rate"][0], 0, 0, 0, hp["learning rate"][0],
                                hp["n actions"], hp["starting beta"], hp["input dims"], hp["guide"], hp["lidar"], hp["lstm"],
                                hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                                hp["mem size"], hp["batch size"], hp["replace"], hp['nstep'], hp["nstep N"], hp["prioritized"],
                                algo='DDQNAgent_centralized_turn', env_name=model_name, chkpt_dir=load_models_path)
            elif hp["agent type"] == "DRQN":
                agent = DRQNAgent(hp["encoding"], hp["number of drones"], hp["discount rate"][0], 0, 0, 0, hp["learning rate"][0],
                                hp["n actions"], hp["starting beta"], hp["input dims"], hp["guide"], hp["lidar"], hp["lstm"],
                                hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                                hp["mem size"], hp["batch size"], hp["replace"], hp['nstep'], hp["nstep N"], hp["prioritized"],
                                algo='DRQNAgent_centralized_turn', env_name=model_name, chkpt_dir=load_models_path)
            elif hp["agent type"] == "DuelDQN":
                agent = DuelDQNAgent(hp["encoding"], hp["number of drones"], hp["discount rate"][0], 0, 0, 0, hp["learning rate"][0],
                                hp["n actions"], hp["starting beta"], hp["input dims"], hp["guide"], hp["lidar"], hp["lstm"],
                                hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                                hp["mem size"], hp["batch size"], hp["replace"], hp['nstep'], hp["nstep N"], hp["prioritized"],
                                algo='DuelDQNAgent_centralized_turn', env_name=model_name, chkpt_dir=load_models_path)
            elif hp["agent type"] == "DuelDDQN":
                agent = DuelDDQNAgent(hp["encoding"], hp["number of drones"], hp["discount rate"][0], 0, 0, 0, hp["learning rate"][0],
                                hp["n actions"], hp["starting beta"], hp["input dims"], hp["guide"], hp["lidar"], hp["lstm"],
                                hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                                hp["mem size"], hp["batch size"], hp["replace"], hp['nstep'], hp["nstep N"], hp["prioritized"],
                                algo='DuelDDQNAgent_centralized_turn', env_name=model_name, chkpt_dir=load_models_path)

            
            hp["curriculum learning"] = {"sparse reward": False, "collisions": False}
            checkpoint = agent.load_models()
            agent.q_eval.eval()
            agent.q_next.eval()
                            #nr, obstacles, set_obstacles, obstacle_density, obstacle_path,\
                # reward_system, positive_reward, negative_reward, positive_exploration_reward, negative_step_reward, \
                # training_type, encoding, guide, lidar, refuel, curriculum_learning, total_episodes, total_steps, exp, ts, save_path, goal_spawning=False
            env = Environment(hp["number of drones"], hp["obstacles"], hp["set obstacles"], hp["obstacle density"], load_path,
                            hp["reward system"], hp["positive rewards"][0], hp["negative rewards"][0],
                            hp["positive exploration rewards"][0], hp["negative step rewards"][0],
                            hp["training type"], hp["encoding"], hp["guide"], hp["lidar"], hp["fuel"],
                            hp["curriculum learning"], 50000, 200, policy_num, 1, load_path, goal_spawning)

            PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])

            i = 0
            fig,ax = plt.subplots(figsize=(WIDTH*2*2, HEIGHT*2))
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
            obstacles = []
            starting_positions = []
            path = []
            paths = []
            traj_actions = []
            backtracking = 0
            steps_taken = 0
            successes = []

            if test == "grid":
                if not hp["number of drones"] > 1:
                    for x in range(WIDTH):
                        for y in range(HEIGHT):
                            # if x == 0 or x == WIDTH-1:
                            #     continue
                            # if y == 0 or y == HEIGHT-1:
                            #     continue
                            image_observation, non_image_observation = env.reset(10000, 99)

                            for i in range(hp["number of drones"]): env.exploration_grid[env.starting_pos[i].y, env.starting_pos[i].x] = False

                            obstacles.append(env.exploration_grid.copy())

                            env.grid[env.pos[0].y, env.pos[0].x] = States.UNEXP.value
                            env.exploration_grid[env.pos[0].y, env.pos[0].x] = False

                            env.starting_pos[0] = Point(x,y)
                            env.pos[0] = Point(x,y)
                            
                            env.grid[env.starting_pos[0].y, env.starting_pos[0].x] = States.ROBOT.value

                            image_observation, non_image_observation, _ = env.get_state()

                            for i in range(1, hp["number of drones"]):
                                env.grid[env.pos[i].y, env.pos[i].x] = States.UNEXP.value
                                env.exploration_grid[env.pos[i].y, env.pos[i].x] = False

                                indices = np.argwhere(env.grid == States.UNEXP.value)
                                np.random.shuffle(indices)
                                env.starting_pos[i] = Point(indices[0,1], indices[0,0])
                                env.pos[i] = Point(indices[0,1], indices[0,0])
                                
                                env.grid[env.starting_pos[i].y, env.starting_pos[i].x] = States.ROBOT.value

                            if hp["stacked frames"]:
                                # adds dimension for previous time steps
                                image_observations = np.expand_dims(image_observation, axis=1)
                                image_observations = np.repeat(image_observations, 4, axis=1)
                                image_observations = image_observations[:, :, 0, :]

                                image_observations_ = np.expand_dims(image_observation, axis=1)
                                image_observations_ = np.repeat(image_observations_, 4, axis=1)
                                image_observations_ = image_observations_[:, :, 0, :]

                            trajectory = []
                            t_actions = []

                            done = False
                            starting_positions.append(list(env.starting_pos))

                            for step in range(int(hp["max steps"][0])):
                                steps_taken += 1
                                path.append(list(env.pos))
                                actions = []
                                action = [None]*hp["number of drones"]
                                for i_r in range(0,hp["number of drones"]):
                                    if hp["stacked frames"]:
                                        image_observations[i_r] = agent.memory.preprocess_observation(step, image_observation)
                                        if step == 0: action[i_r] = agent.choose_action(env, i_r, image_observations[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], action)
                                        else: action[i_r] = agent.choose_action(env, i_r, image_observations[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], action, previous_action[i_r])
                                    else:
                                        if step == 0: action[i_r], _ = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], hp["stacked frames"], action)
                                        else: action[i_r], _ = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], hp["stacked frames"], action, previous_action[i_r])
                                    
                                    previous_action = action.copy()
                                actions.append(action)

                                for r_i in range(hp["number of drones"]):
                                    if action[r_i] == Direction.LEFT.value:
                                        if env.grid[env.pos[r_i].y][env.pos[r_i].x-1] == States.EXP.value: backtracking += 1
                                    if action[r_i] == Direction.RIGHT.value:
                                        if env.grid[env.pos[r_i].y][env.pos[r_i].x+1] == States.EXP.value: backtracking += 1
                                    if action[r_i] == Direction.UP.value:
                                        if env.grid[env.pos[r_i].y-1][env.pos[r_i].x] == States.EXP.value: backtracking += 1
                                    if action[r_i] == Direction.DOWN.value:
                                        if env.grid[env.pos[r_i].y+1][env.pos[r_i].x] == States.EXP.value: backtracking += 1

                                image_observation_, non_image_observation_, reward, done, info = env.step_centralized(action)
                                cnt += info[0]
                                t_actions.append(action)

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
                                    successes.append(info[0])
                                    paths.append(path)
                                    traj_actions.append(t_actions)
                                    path = []
                                    t_actions = []

                                if done:
                                    successes.append(info[0])
                                    paths.append(path)
                                    traj_actions.append(t_actions)
                                    path = []
                                    t_actions = []
                                    if show_plot:
                                        plt.cla()
                                        PR.print_trajectories(ax, save_path, policy, env, actions[0], reward, done)
                                        if save_path:
                                            for i in range(hp["number of drones"]):
                                                PR.print_graph(False, policy, i, path, 
                                                t_actions, 
                                                env.starting_pos, env.exploration_grid.copy(), 
                                                load_path, len(paths), env)
                                    trajectories.append(trajectory)
                                    break
                                if save_plot and len(paths) < 2:
                                    for i in range(hp["number of drones"]):
                                        PR.print_graph(False, policy, i, [sub_path[i] for sub_path in path], 
                                        [sub_actions[i] for sub_actions in t_actions], 
                                        env.starting_pos[i], env.exploration_grid.copy(), 
                                        load_path, len(paths), env, True, step)
                                if show_plot:
                                    plt.cla()
                                    PR.print_trajectories(ax, save_path, policy, env, actions[0])
                                    if save_plot:
                                        file_name = "policy%d_trajectory%d_step%d.png" %(policy, i, step)
                                        plt.savefig(os.path.join(save_path, file_name))
                            steps.append(step)
                            # if step == int(ms)-1 and not done:
                            #     trajectories.append(trajectory)
                        path = []
                        t_actions = []
                    
                    if save_plot:
                        for i in range(hp["number of drones"]):
                            for cntr, path in enumerate(paths):
                                PR.print_graph(successes[cntr], policy, i, [sub_path[i] for sub_path in path], 
                                            [sub_actions[i] for sub_actions in traj_actions[cntr]], 
                                            [sub_starting_position[i] for sub_starting_position in starting_positions][cntr], obstacles[cntr], 
                                            load_path, cntr, env)

                    p = cnt/((WIDTH)*(HEIGHT))*100
                    print("Percentage success: %d / %d x 100 = %.2f %%" %(cnt, (WIDTH)*(HEIGHT), p))
                    print("Average steps: %.2f" %(np.mean(np.array(steps))))
                    print("Average backtracking: %.2f" %(backtracking/steps_taken))

                    file_name = "Results%s_%s.json" %(str(policy), str(ts))
                    file_name = os.path.join(load_path, file_name)
                    write_json("Success:%s, Average steps:%s, Average collisions:%s, Average timeouts:%s" %(str(p), str(np.mean(np.array(steps))), str(np.mean(collisions_grid)), str(timeout_cntr/testing_iterations)), file_name)


                    print(collisions_grid)
                    print("Average collisions: ", np.mean(collisions_grid))

                    print("Timed out: ", timeout_cntr)
                else:
                    stop = 601
                    for x1 in range(WIDTH):
                        if x1 != 0: print(x1, "\n", cnt/(len(paths))*100)
                        for y1 in range(HEIGHT):
                            for x2 in range(WIDTH):
                                for y2 in range(HEIGHT):
                                    if y2 == y1 and x2 == x1: continue
                                    if len(paths) == stop:
                                        breakpoint

                                    if env.grid[y1,x1] == States.OBS.value or env.grid[y2,x2] == States.OBS.value:
                                        continue
                                    # if x == 0 or x == WIDTH-1:
                                    #     continue
                                    # if y == 0 or y == HEIGHT-1:
                                    #     continue
                                    image_observation, non_image_observation = env.reset(10000, 99)

                                    for i in range(hp["number of drones"]): env.exploration_grid[env.starting_pos[i].y, env.starting_pos[i].x] = False

                                    obstacles.append(env.exploration_grid.copy())

                                    env.grid[env.pos[0].y, env.pos[0].x] = States.UNEXP.value
                                    env.exploration_grid[env.pos[0].y, env.pos[0].x] = False

                                    env.starting_pos[0] = Point(x1,y1)
                                    env.pos[0] = Point(x1,y1)
                                    
                                    env.grid[env.starting_pos[0].y, env.starting_pos[0].x] = States.ROBOT.value

                                    env.grid[env.pos[1].y, env.pos[1].x] = States.UNEXP.value
                                    env.exploration_grid[env.pos[1].y, env.pos[1].x] = False

                                    env.starting_pos[1] = Point(x2,y2)
                                    env.pos[1] = Point(x2,y2)
                                    
                                    env.grid[env.starting_pos[1].y, env.starting_pos[1].x] = States.ROBOT.value

                                    for i in range(hp["number of drones"]): env.exploration_grid[env.starting_pos[i].y, env.starting_pos[i].x] = True

                                    if hp["n actions"] == 3:
                                        for r in range(env.nr):
                                            env.prev_pos[r] = Point(env.starting_pos[r].x-1, env.starting_pos[r].y)
                                    
                                    # if env.starting_pos[0] == Point(2,0) and env.starting_pos[1] == Point(1,0):
                                    #     env.grid[env.goal.y, env.goal.x] = States.UNEXP.value
                                    #     env.goal = Point(3,2)
                                    #     env.grid[env.goal.y, env.goal.x] = States.GOAL.value
                                    
                                    image_observation, non_image_observation, closest_unexplored = env.get_state()
                                    env.prev_closest_unexplored = closest_unexplored.copy()

                                    if hp["stacked frames"]:
                                        # adds dimension for previous time steps
                                        image_observations = np.expand_dims(image_observation, axis=1)
                                        image_observations = np.repeat(image_observations, 4, axis=1)
                                        image_observations = image_observations[:, :, 0, :]

                                        image_observations_ = np.expand_dims(image_observation, axis=1)
                                        image_observations_ = np.repeat(image_observations_, 4, axis=1)
                                        image_observations_ = image_observations_[:, :, 0, :]

                                    trajectory = []
                                    t_actions = []

                                    done = False
                                    starting_positions.append(list(env.starting_pos))

                                    for step in range(int(hp["max steps"][0])):
                                        steps_taken += 1
                                        path.append(list(env.pos))
                                        actions = []
                                        action = [None]*hp["number of drones"]
                                        for i_r in range(0,hp["number of drones"]):
                                            if hp["stacked frames"]:
                                                image_observations[i_r] = agent.memory.preprocess_observation(step, image_observation)
                                                if step == 0: action[i_r] = agent.choose_action(env, i_r, image_observations[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], action)
                                                else: action[i_r] = agent.choose_action(env, i_r, image_observations[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], action, previous_action[i_r])
                                            else:
                                                if step == 0: action[i_r], _ = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], hp["stacked frames"], action)
                                                else: action[i_r], _ = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], hp["stacked frames"], action, previous_action[i_r])
                                            
                                            previous_action = action.copy()
                                        actions.append(action)

                                        # for r_i in range(hp["number of drones"]):
                                        #     if action[r_i] == Direction.LEFT.value:
                                        #         if env.grid[env.pos[r_i].y][env.pos[r_i].x-1] == States.EXP.value: backtracking += 1
                                        #     if action[r_i] == Direction.RIGHT.value:
                                        #         if env.grid[env.pos[r_i].y][env.pos[r_i].x+1] == States.EXP.value: backtracking += 1
                                        #     if action[r_i] == Direction.UP.value:
                                        #         if env.grid[env.pos[r_i].y-1][env.pos[r_i].x] == States.EXP.value: backtracking += 1
                                        #     if action[r_i] == Direction.DOWN.value:
                                        #         if env.grid[env.pos[r_i].y+1][env.pos[r_i].x] == States.EXP.value: backtracking += 1

                                        image_observation_, non_image_observation_, reward, done, info = env.step_centralized(action)
                                        cnt += info[0]
                                        t_actions.append(action)

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
                                            successes.append(info[0])
                                            paths.append(path)
                                            traj_actions.append(t_actions)
                                            path = []
                                            t_actions = []

                                        if done:
                                            successes.append(info[0])
                                            paths.append(path)
                                            traj_actions.append(t_actions)
                                            path = []
                                            t_actions = []
                                            if show_plot:
                                                plt.cla()
                                                PR.print_trajectories(ax, save_path, policy, env, actions[0], reward, done)
                                                # if save_path:
                                                #     for i in range(hp["number of drones"]):
                                                #         PR.print_graph(False, policy, i, path, 
                                                #         t_actions, 
                                                #         env.starting_pos, env.exploration_grid.copy(), 
                                                #         load_path, len(paths), env)
                                            trajectories.append(trajectory)
                                            break
                                        # if save_path and len(paths) < 2:
                                        #     for i in range(hp["number of drones"]):
                                        #         PR.print_graph(False, policy, i, [sub_path[i] for sub_path in path], 
                                        #         [sub_actions[i] for sub_actions in t_actions], 
                                        #         env.starting_pos[i], env.exploration_grid.copy(), 
                                        #         load_path, len(paths), env, True, step)
                                        if show_plot:
                                            plt.cla()
                                            PR.print_trajectories(ax, save_path, policy, env, actions[0])
                                            if save_plot:
                                                file_name = "policy%d_trajectory%d_step%d.png" %(policy, i, step)
                                                plt.savefig(os.path.join(save_path, file_name))
                                    steps.append(step)
                                    # if step == int(ms)-1 and not done:
                                    #     trajectories.append(trajectory)
                                path = []
                                t_actions = []
                            
                    if save_plot:
                        for cntr, path in enumerate(paths):
                            for i in range(hp["number of drones"]):
                                if [sub_starting_position[0] for sub_starting_position in starting_positions][cntr] == Point(2,0) and [sub_starting_position[1] for sub_starting_position in starting_positions][cntr] == Point(1,0):
                                    PR.print_graph(successes[cntr], policy, i, [sub_path[i] for sub_path in path], 
                                                [sub_actions[i] for sub_actions in traj_actions[cntr]], 
                                                [sub_starting_position[i] for sub_starting_position in starting_positions][cntr], obstacles[cntr], 
                                                load_path, cntr, env)

                    p = cnt/(len(paths))*100
                    print("Percentage success: %d / %d x 100 = %.2f %%" %(cnt, len(paths), p))
                    print("Average steps: %.2f" %(np.mean(np.array(steps))))
                    print("Average backtracking: %.2f" %(backtracking/steps_taken))

                    file_name = "Results%s_%s.json" %(str(policy), str(ts))
                    file_name = os.path.join(load_path, file_name)
                    write_json("Success:%s, Average steps:%s, Average collisions:%s, Average timeouts:%s" %(str(p), str(np.mean(np.array(steps))), str(np.mean(collisions_grid)), str(timeout_cntr/testing_iterations)), file_name)


                    print(collisions_grid)
                    print("Average collisions: ", np.mean(collisions_grid))

                    print("Timed out: ", timeout_cntr)
            elif test == "iterative":
                for i in range(0, testing_iterations):
                    if i % 10 == 0 and i != 0:
                        print("%d: %.2f %%, %.2f steps" %(int(i), float(cnt)/float(i)*100, np.mean(np.array(steps))))
                    image_observation, non_image_observation = env.reset(10000, 99)

                    env.exploration_grid[env.starting_pos[0].y, env.starting_pos[0].x] = False

                    obstacles.append(env.exploration_grid.copy())

                    env.exploration_grid[env.starting_pos[0].y, env.starting_pos[0].x] = True

                    if hp["stacked frames"]:
                        # adds dimension for previous time steps
                        image_observations = np.expand_dims(image_observation, axis=1)
                        image_observations = np.repeat(image_observations, 4, axis=1)
                        image_observations = image_observations[:, :, 0, :]

                        image_observations_ = np.expand_dims(image_observation, axis=1)
                        image_observations_ = np.repeat(image_observations_, 4, axis=1)
                        image_observations_ = image_observations_[:, :, 0, :]

                    trajectory = []
                    t_actions = []

                    done = False
                    
                    starting_positions.append(env.starting_pos[0])

                    for step in range(int(hp["max steps"][0])):
                        steps_taken += 1
                        for i_r in range(0,hp["number of drones"]):
                            path.append(env.pos[i_r])
                        actions = []
                        action = [0]*hp["number of drones"]
                        for i_r in range(0,hp["number of drones"]):
                            if hp["stacked frames"]:
                                image_observations[i_r] = agent.memory.preprocess_observation(step, image_observation)
                                if step == 0: action[i_r] = agent.choose_action(env, i_r, image_observations[i_r], non_image_observation[i_r], hp["allow windowed revisiting"])
                                else: action[i_r] = agent.choose_action(env, i_r, image_observations[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], previous_action[i_r])
                            else:
                                # if step == 0: action[i_r] = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], hp["stacked frames"], action)
                                # else: action[i_r] = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], hp["stacked frames"], action, previous_action[i_r])
                                if step == 0: action[i_r], _ = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], hp["stacked frames"], action)
                                else: action[i_r], _ = agent.choose_action(env, i_r, image_observation[i_r], non_image_observation[i_r], hp["allow windowed revisiting"], hp["stacked frames"], action, previous_action[i_r])
                            
                            previous_action = action.copy()
                        actions.append(action)

                        # for r_i in range(hp["number of drones"]):
                        #     if action[r_i] == Direction.LEFT.value:
                        #         if env.grid[env.pos[r_i].y][env.pos[r_i].x-1] == States.EXP.value: backtracking += 1
                        #     if action[r_i] == Direction.RIGHT.value:
                        #         if env.grid[env.pos[r_i].y][env.pos[r_i].x+1] == States.EXP.value: backtracking += 1
                        #     if action[r_i] == Direction.UP.value:
                        #         if env.grid[env.pos[r_i].y-1][env.pos[r_i].x] == States.EXP.value: backtracking += 1
                        #     if action[r_i] == Direction.DOWN.value:
                        #         if env.grid[env.pos[r_i].y+1][env.pos[r_i].x] == States.EXP.value: backtracking += 1

                        # trajectory.append((env.pos[i_r], action, i_r))
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
                            PR.print_trajectories(ax, save_path, policy, env, actions[0])
                            if save_plot:
                                file_name = "p%dtrajectory%d%d.png" %(policy, i, step)
                                plt.savefig(os.path.join(save_path, file_name))
                    steps.append(step)

                p = cnt/(testing_iterations)*100
                average_backtracking = backtracking/steps_taken*100 
                print("Percentage success: %d / %d x 100 = %.2f %%" %(cnt, testing_iterations, p))
                print("Average steps: %.2f" %(np.mean(np.array(steps))))
                print("Average backtracking: %.2f" %(average_backtracking))

                if goal_spawning:
                    file_name = "Results%s_%s_goal.json" %(str(policy), str(ts))
                else:
                    file_name = "Results%s_%s.json" %(str(policy), str(ts))
                file_name = os.path.join(load_path, file_name)
                write_json("Success:%s, Average steps:%s, Average collisions:%s, Average timeouts:%s Average backtracking:%s" \
                        %(str(p), str(np.mean(np.array(steps))), \
                            str(np.mean(collisions_grid)), \
                                str(timeout_cntr/testing_iterations), \
                                    str(average_backtracking)), \
                                        file_name)


                print(collisions_grid)
                print("Average collisions: ", np.mean(collisions_grid))

                print("Timed out: ", timeout_cntr)
