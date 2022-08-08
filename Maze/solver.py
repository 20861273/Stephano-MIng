from dijkstra import run_dijkstra
from qlearning import run_qlearning
from astar import Astar


method = input("With which method would you like to solve the maze?\n1. Q-learning\n2. DQN\n3. Dijkstra's algorithm\n4. A* algorithm\nSelect method:")
method_flag = int(method)

if method_flag == 1:    # Q-learning
    debug_q = input("\nTraining mode: 1\nGreedy policy: 2\nGenerated map: 3\nSelect mode: ")
    debug_flag = int(debug_q)
    run_qlearning(debug_flag)
elif method_flag == 2:  # NN
    pass
elif method_flag == 3:  # Dijkstra's algorithm
    debug_q = input("\nDefault mode: 1\nGenerated map: 2\nSelect mode: ")
    debug_flag = int(debug_q)
    run_dijkstra(debug_flag)
elif method_flag == 4:  # A* algorithm
    debug_q = input("\nDefault mode: 1\nGenerated map: 2\nSelect mode: ")
    debug_flag = int(debug_q)
    astar = Astar()
    astar.run_astar(debug_flag)