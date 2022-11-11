from dec_m_qlearning import QLearning
from dec_m_environment import Environment

method = input("With which method would you like to solve the SAR problem?\n1. Q-learning\n2. DQN\n3. A* algorithm\n4. Lawnmower pattern\n\nSelect method:")
method_flag = int(method)

if method_flag == 1:    # Q-learning
    debug_q = input("\n1. Training mode\n2. Greedy policy\n3. Generated map\nSelect mode: ")
    debug_flag = int(debug_q)
    env = Environment(2)
    ql = QLearning(env)
    ql.run_qlearning(debug_flag)
elif method_flag == 2:  # NN
    pass
elif method_flag == 3:  # A* algorithm
    pass
elif method_flag == 4: # Lawnmower pattern
    pass
    