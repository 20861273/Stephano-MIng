from dijkstra import run_dijkstra
from qlearning import run_qlearning


method = input("With which method would you like to solve the maze?\n1. Q-learning\n2. DQN\n3. Dijkstra's algorithm\n4. A* algorithm\nSelect method:")
method_flag = int(method)

if method_flag == 1:
    run_qlearning()
elif method_flag == 2:
    pass
elif method_flag == 3:
    run_dijkstra()
elif method_flag == 4:
    pass