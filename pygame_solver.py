import os
import json
from re import M

from pygame_environment import Environment
from pygame_qlearning import QLearning


PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')

input_file = open(os.path.join(PATH, "input.json"))
input_data = json.load(input_file)

method = input("With which method would you like to solve the SAR problem?\n1. Q-learning\n2. DQN\n3. A* algorithm\n4. Lawnmower pattern\n\nSelect method:")
method_flag = int(method)

if method_flag == 1:    # Q-learning
    debug_q = input("\n1. Training mode\n2. Greedy policy\n3. Generated map\nSelect mode: ")
    debug_flag = int(debug_q)
    
    environment = Environment(input_data)
    ql = QLearning(environment, input_data)
    ql.run_qlearning(debug_flag, environment)
    input_file.close()
elif method_flag == 2:  # NN
    pass
elif method_flag == 3:  # A* algorithm
    pass
elif method_flag == 4: # Lawnmower pattern
    pass


    