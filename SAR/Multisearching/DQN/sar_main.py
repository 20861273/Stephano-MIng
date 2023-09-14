from dqn_utils import plot_learning_curve, write_json, read_json, save_hp
from dqn_environment import Environment, HEIGHT, WIDTH, Point, States, Direction
from datetime import datetime
import os
import torch as T
from dqn_save_results import print_results
import matplotlib.pyplot as plt
from dqn_decentralized_testing import test_dqn
from dqn_distributed_training import distributed_dqn
from dqn_centralized_training import centralized_dqn
from dqn_centralized_testing import test_centralized_dqn
from dqn_test_training import test_dqn
from dqn_agent import DQNAgent

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    testing_parameters = {
                        "training": True,
                        "load checkpoint": False,
                        "show rewards interval": 100,
                        "show plot": False,
                        "save plot": False,
                        "policy number": [0],
                        "session": [],
                        "test type": "grid", # test types: grid and iterative
                        "testing iterations": 500,
                        "goal spawning": False
    }

    # encodings: image (n_images, H, W), image_occupancy (n_images, H, W), full_image (H, W), position (H*W), position_exploration (H*W*2), position_occupancy (H*W*2)
    # agent types: DQN, DDQN
    hp = {
                    "number of drones": 4,
                    "training type": "centralized", # centralized (turn based), centralized actions
                    "agent type": "DQN",
                    "learning rate": [0.0001],
                    "discount rate": [0.75,0.8,0.85],
                    "epsilon": [[1,0.1,0.1]],

                    "training sessions": 3,
                    "episodes": 10000,
                    "positive rewards": [0],
                    "positive exploration rewards": [1],
                    "negative rewards": [1],
                    "negative step rewards": [0.05],
                    "max steps": [200],

                    "n actions": 4,
                    "env size": '%sx%s' %(str(WIDTH), str(HEIGHT)),
                    "obstacles": False,
                    "obstacle density": 0.3,
                    "encoding": "local",
                    "stacked frames": False,
                    "input dims": (2,HEIGHT, WIDTH),
                    "lidar": False,
                    "guide": False,
                    "fuel": False,
                    "lstm": False,

                    "batch size": 64,
                    "mem size": 100000,
                    "replace": 1000,
                    "channels": [32, 64],
                    "kernel": [2, 2],
                    "stride": [1, 1],
                    "fc dims": [16,32],

                    "nstep": False,
                    "nstep N": 10,
                    "prioritized": True,
                    "starting beta": 0.5,

                    "device": 0,

                    "allow windowed revisiting": True,
                    "curriculum learning": {"sparse reward": False, "collisions": False},
                    "reward system": {"find goal": False, "coverage": True},
    }
    
    if hp["number of drones"] < 2 and hp["training type"] == "decentralized":
        print("Cannot have less than 2 drones and train decentralized...")
        quit()
    if hp["number of drones"] < 2 and not hp["input dims"][0] == 2:
        print("Input dimensions error...")
        quit()

    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'SAR')
    PATH = os.path.join(PATH, 'Results')
    PATH = os.path.join(PATH, 'DQN')
    load_path = os.path.join(PATH, 'Saved_data')
    if not os.path.exists(load_path): os.makedirs(load_path)        

    load_checkpoint_path = os.path.join(PATH, "Saved_data")
    if testing_parameters["load checkpoint"]:
        save_path = load_checkpoint_path
        models_path = os.path.join(save_path, 'models')
    else:
        date_and_time = datetime.now()
        save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
        if not os.path.exists(save_path): os.makedirs(save_path)
        models_path = os.path.join(save_path, 'models')
        if not os.path.exists(models_path): os.makedirs(models_path)

    env_size = '%sx%s' %(str(WIDTH), str(HEIGHT))

    if testing_parameters["load checkpoint"] and testing_parameters["training"]:
        file_name = "hyperparameters.json"
        file_name = os.path.join(load_checkpoint_path, file_name)
        hp = read_json(file_name)
        checkpoint = {'session':os.listdir(models_path)[-1][0]}
        outliers = [[Point(0,2),Point(3,4)],[Point(1,0),Point(2,4)],[Point(2,4),Point(1,0)],[Point(3,4),Point(0,2)],[Point(5,5),Point(5,1)]]
    else:
        checkpoint = {'session':0}
        outliers = None

    num_experiences =     len(hp["learning rate"]) \
                        * len(hp["discount rate"]) \
                        * len(hp["epsilon"]) \
                        * len(hp["positive rewards"]) \
                        * len(hp["negative rewards"]) \
                        * len(hp["positive exploration rewards"]) \
                        * len(hp["negative step rewards"]) \
                        * len(hp["max steps"]) \
                        * hp["training sessions"]
    
    # set input dimensions to encoding type
    if hp["encoding"] == "position":
        hp["input dims"] = [HEIGHT*WIDTH]
    elif hp["encoding"] == "position_exploration" or hp["encoding"] == "position_occupancy":
        hp["input dims"] = [HEIGHT*WIDTH*2]
    elif hp["encoding"] == "image" or hp["encoding"] == "image_occupancy":
        if hp["stacked frames"]:
            hp["input dims"] = (4,3,HEIGHT, WIDTH)
            if hp["number of drones"] > 1: hp["input dims"] = (4,4,HEIGHT, WIDTH)
        else:
            hp["input dims"] = (2,HEIGHT, WIDTH)
            if hp["obstacles"]: hp["input dims"] = (3,HEIGHT, WIDTH)
            if hp["number of drones"] > 1:
                # hp["input dims"] = (4,HEIGHT, WIDTH)
                # hp["input dims"] = (3,HEIGHT*10, WIDTH*10)
                hp["input dims"] = (3,HEIGHT, WIDTH)
                if hp["obstacles"]: hp["input dims"] = (4,HEIGHT, WIDTH)
            # hp["input dims"] = (1,HEIGHT, WIDTH)
            # if hp["obstacles"]: hp["input dims"] = (2,HEIGHT, WIDTH)
            # if hp["number of drones"] > 1:
            #     # hp["input dims"] = (4,HEIGHT, WIDTH)
            #     hp["input dims"] = (2,HEIGHT, WIDTH)
            #     if hp["obstacles"]: hp["input dims"] = (3,HEIGHT, WIDTH)
    elif hp["encoding"] == "full_image":
        if hp["stacked frames"]: hp["input dims"] = (4,HEIGHT, WIDTH)
        else: hp["input dims"] = (1,HEIGHT, WIDTH)
    elif hp["encoding"] == "local":
        hp["input dims"] = (4)#+2*(hp["number of drones"]-1))

    if hp["lidar"] and "image" not in hp["encoding"]:
        hp["input dims"][0] += 4
    
    if hp["agent type"] == "DRQN": hp["lstm"] = True

    if not testing_parameters["load checkpoint"]:
        save_hp(save_path, hp)

    if testing_parameters["training"]:
        print("Number of training sessoins: ", num_experiences)
        i_exp = 0
        # training session loops
        for pr_i in hp["positive rewards"]:
            for nr_i in hp["negative rewards"]:
                for per_i in hp["positive exploration rewards"]:
                    for nsr_i in hp["negative step rewards"]:
                        for ms_i in hp["max steps"]:
                            for lr_i in hp["learning rate"]:
                                for dr_i in hp["discount rate"]:
                                    for er_i in hp["epsilon"]:
                                        if i_exp >= int(checkpoint['session']):
                                            if hp["training type"] == "centralized":
                                                load_checkpoint = centralized_dqn(
                                                            hp["number of drones"], hp["obstacles"], hp["obstacle density"] ,hp["training sessions"], hp["episodes"], testing_parameters["show rewards interval"], hp["training type"], hp["agent type"], hp["encoding"],
                                                            hp["curriculum learning"], hp["reward system"], hp["allow windowed revisiting"],
                                                            dr_i, lr_i, er_i[0], er_i[1], er_i[2],
                                                            pr_i, nr_i, per_i, nsr_i, ms_i, i_exp,
                                                            hp["n actions"], hp["starting beta"], hp["input dims"], hp["stacked frames"], hp["guide"], hp["lidar"], hp["fuel"],
                                                            hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                                                            hp["batch size"], hp["mem size"], hp["replace"], hp["nstep"], hp["nstep N"],
                                                            hp["prioritized"], models_path, save_path, load_checkpoint_path, hp["env size"], testing_parameters["load checkpoint"], hp["device"], hp["lstm"],
                                                            outliers)
                                            elif hp["training type"] == "centralized actions":
                                                load_checkpoint = distributed_dqn(
                                                            hp["number of drones"], hp["training sessions"], hp["episodes"], testing_parameters["show rewards interval"], hp["training type"], hp["encoding"],
                                                            dr_i, lr_i, er_i[0], er_i[1], er_i[2],
                                                            pr_i, nr_i, per_i, nsr_i, ms_i, i_exp,
                                                            hp["n actions"], hp["starting beta"], hp["input dims"],
                                                            hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                                                            hp["batch size"], hp["mem size"], hp["replace"],
                                                            hp["prioritized"], models_path, save_path, load_checkpoint_path, hp["env size"], testing_parameters["load checkpoint"])
                                            elif hp["training type"] == "test":
                                                load_checkpoint = test_dqn(
                                                            hp["number of drones"], hp["training sessions"], hp["episodes"], testing_parameters["show rewards interval"], hp["training type"], hp["encoding"],
                                                            dr_i, lr_i, er_i[0], er_i[1], er_i[2],
                                                            pr_i, nr_i, per_i, nsr_i, ms_i, i_exp,
                                                            hp["n actions"], hp["starting beta"], hp["input dims"],
                                                            hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                                                            hp["batch size"], hp["mem size"], hp["replace"],
                                                            hp["prioritized"], models_path, load_path, save_path, load_checkpoint_path, hp["env size"], testing_parameters["load checkpoint"])
                                        i_exp += 1
    else:
        if hp["training type"] == "centralized":
            test_centralized_dqn(testing_parameters["policy number"], testing_parameters["session"], load_path, save_path, models_path,
                                 testing_parameters["testing iterations"], testing_parameters["goal spawning"], testing_parameters["show plot"],
                                 testing_parameters["save plot"], testing_parameters["test type"])

