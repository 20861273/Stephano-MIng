from qlearning import QLearning

method = input("With which method would you like to solve the maze?\n1. Q-learning\n2. DQN\n3. A* algorithm\n4. Lawnmower pattern\n\nSelect method:")
method_flag = int(method)

if method_flag == 1:    # Q-learning
    debug_q = input("\nTraining mode: 1\nGreedy policy: 2\nGenerated map: 3\nSelect mode: ")
    debug_flag = int(debug_q)
    ql = QLearning()
    ql.run_qlearning(debug_flag)
elif method_flag == 2:  # NN
    pass
elif method_flag == 3:  # Dijkstra's algorithm
    pass
elif method_flag == 4:  # A* algorithm
    debug_q = input("\nDefault mode: 1\nGenerated map: 2\nSelect mode: ")
    debug_flag = int(debug_q)
    