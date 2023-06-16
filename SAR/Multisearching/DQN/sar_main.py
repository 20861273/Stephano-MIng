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
                        "show rewards interval": 1000,
                        "show plot": True,
                        "save plot": False,
                        "policy number": [0],
                        "testing iterations": 10000
    }

    hp = {
                    "number of drones": 1,
                    "training type": "centralized",
                    "learning rate": [0.0001],
                    "discount rate": [0.5,0.7,0.9,0.99],
                    "epsilon": [[0.01,0.01,0.01]],

                    "training sessions": 1,
                    "episodes": 50000,
                    "positive rewards": [1],
                    "positive exploration rewards": [0],
                    "negative rewards": [1],
                    "negative step rewards": [0.01],
                    "max steps": [200],

                    "n actions": 4,
                    "env size": '%sx%s' %(str(WIDTH), str(HEIGHT)),
                    "encoding": "image",
                    "input dims": (2,HEIGHT, WIDTH),
                    "lidar": True,

                    "batch size": 64,
                    "mem size": 100000,
                    "replace": 100,
                    "channels": [16, 32],
                    "kernel": [2, 2],
                    "stride": [1, 1],
                    "fc dims": [32],

                    "prioritized": True,
                    "starting beta": 0.5,

                    "device": 0,

                    "allow windowed revisiting": True,
                    "curriculum learning": {"sparse reward": True, "collisions": False},
                    "reward system": {"find goal": True, "coverage": False}
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

    load_checkpoint_path = os.path.join(PATH, "13-06-2023 11h58m33s")
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
    
    if not testing_parameters["load checkpoint"]:
        save_hp(save_path, hp)

    if testing_parameters["load checkpoint"] and testing_parameters["training"]:
        file_name = "hyperparameters.json"
        file_name = os.path.join(load_checkpoint_path, file_name)
        hp = read_json(file_name)
        checkpoint = {'session':os.listdir(models_path)[-1][0]}
    else:
        checkpoint = {'session':0}

    num_experiences =     len(hp["learning rate"]) \
                        * len(hp["discount rate"]) \
                        * len(hp["epsilon"]) \
                        * len(hp["positive rewards"]) \
                        * len(hp["negative rewards"]) \
                        * len(hp["positive exploration rewards"]) \
                        * len(hp["negative step rewards"]) \
                        * len(hp["max steps"]) \
                        * hp["training sessions"]
    
    if hp["encoding"] == "position":
        hp["input dims"] = [HEIGHT*WIDTH]
    elif hp["encoding"] == "position_exploration":
        hp["input dims"] = [HEIGHT*WIDTH*2]

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
                                                            hp["number of drones"], hp["training sessions"], hp["episodes"], testing_parameters["show rewards interval"], hp["training type"], hp["encoding"],
                                                            hp["curriculum learning"], hp["reward system"], hp["allow windowed revisiting"],
                                                            dr_i, lr_i, er_i[0], er_i[1], er_i[2],
                                                            pr_i, nr_i, per_i, nsr_i, ms_i, i_exp,
                                                            hp["n actions"], hp["starting beta"], hp["input dims"], hp["lidar"],
                                                            hp["channels"], hp["kernel"], hp["stride"], hp["fc dims"],
                                                            hp["batch size"], hp["mem size"], hp["replace"],
                                                            hp["prioritized"], models_path, save_path, load_checkpoint_path, hp["env size"], testing_parameters["load checkpoint"], hp["device"])
                                            elif hp["training type"] == "decentralized":
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
            test_centralized_dqn(testing_parameters["policy number"], load_path, save_path, models_path, testing_parameters["testing iterations"], testing_parameters["show plot"], testing_parameters["save plot"])

