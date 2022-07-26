from maze import HEIGHT, WIDTH, Maze, States
#from test import HEIGHT, WIDTH, MazeAI, States

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import os

import heapq

def run_dijkstra():
    env = Maze()

    g = Graph(env.grid)
    print("Grid:\n", env.grid)
    print("Starting position: ", env.starting_pos)
    print("Goal: ", env.exit)
    start, goal = (env.starting_pos.x, env.starting_pos.y), (env.exit.x, env.exit.y)
    #print(start, goal, "\n",env.grid)

    came_from, cost_so_far = dijkstra_search(g, start, goal)

    path_taken = reconstruct_path(came_from, start=start, goal=goal)
    #print(path_taken)
    PATH = os.getcwd()
    PATH = os.path.join(PATH, 'Results')
    PATH = os.path.join(PATH, 'Dijkstra')
    date_and_time = datetime.now()
    save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
    if not os.path.exists(save_path): os.makedirs(save_path)

    env.grid[path_taken[0][1], path_taken[0][0]] = States.ROBOT.value
    env.grid[env.exit.y, env.exit.x] = States.EXIT.value

    for i in np.arange(len(path_taken)):
        if i != 0:
            env.grid[path_taken[i-1][1], path_taken[i-1][0]] = States.EXP.value
            env.grid[path_taken[i][1], path_taken[i][0]] = States.ROBOT.value

        PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])
        PR.print_graph(i)
        
        file_name = "plot-%s.png" %(i)
        plt.savefig(os.path.join(save_path, file_name))
        plt.close()

def reset(env):
    # Generates grid
    env.grid = env.generate_grid()
    env.pos = env.starting_pos
    env.grid[env.starting_pos.y, env.starting_pos.x] = States.ROBOT.value
    env.grid[env.exit.y, env.exit.x] = States.EXIT.value

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return not self.elements
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

class Graph:
    def __init__(self, maze):
        self._get_nodes_and_walls(maze)
        self.weights = {}

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < WIDTH and 0 <= y < HEIGHT
    
    def passable(self, id):
        return id not in self.walls

    # Only adds nodes that are movable
    # Also stores walls
    def _get_nodes_and_walls(self, maze):
        self.all_nodes = []
        self.walls = []
        for x in range(len(maze[0])):
            for y in range(len(maze)):
                #print(maze[y,x])
                if maze[y,x] == 1:
                    self.walls.append((x, y))
                else:
                    self.all_nodes.append([x, y])

    # Gets neighbours for each node if the neighbour is a node
    def get_neighbors(self, node):
        (x, y) = node
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] # E W N S
        
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        results = list(filter(self.in_bounds, neighbors))
        results = list(filter(self.passable, results))
        return results
    
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)

def dijkstra_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        for next in graph.get_neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):

    current = goal
    path = []
    while current != start: # note: this will fail if no path found
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

class print_results:
    """
    A class used to print the results
    ...
    Attributes
    ----------
    grid : int
        3D array of grid-based environment at each time step. (grid[time_step, y, x])
    rows : int
        number of rows in the environment
    cols : int
        number of columns in the environment
    n_r : int
        the number of robots
    Methods
    -------
    def print_graph(self):
        prints the grid environment
    """

    def __init__(self,grid,rows,cols):
        self.grid = grid
        self.rows = rows
        self.cols = cols
    def print_graph(self, step):
        """
        Prints the grid environment
        """

        plt.rc('font', size=12)
        plt.rc('axes', titlesize=15) 

        # Prints graph
        fig,ax = plt.subplots(figsize=(8, 8))

        # Set tick locations
        ax.set_xticks(np.arange(-0.5, self.cols*2+0.5, step=2),minor=False)
        ax.set_yticks(np.arange(-0.5, self.rows*2+0.5, step=2),minor=False)
        
        plt.xticks(rotation=90)
    
        xticks = list(map(str,np.arange(0, self.cols+1, step=1)))
        yticks = list(map(str,np.arange(0, self.rows+1, step=1)))
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)

        # Set grid
        plt.grid(which='major',axis='both', color='k')

        # Print
        for j in range(self.rows):
            for i in range(self.cols):
                x1 = (i-0.5)*2 + 0.5
                x2 = (i+0.5)*2 + 0.5
                y1 = (self.rows - (j-0.5) - 1)*2 + 0.5
                y2 = (self.rows - (j+0.5) - 1)*2 + 0.5
                if self.grid[j][i] == 0:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'w', alpha=0.75)
                elif self.grid[j][i] == 1:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'b', alpha=0.75)
                elif self.grid[j][i] == 2:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'g', alpha=0.75)
                elif self.grid[j][i] == 3:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'r', alpha=0.75)
                #elif self.grid[j][i] == 4:
                #    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'b', alpha=0.75)

        plt_title = "Dijkstra's Algorithm Results: Step %s" %(str(step))
        plt.title(plt_title)

