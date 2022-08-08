from math import inf
from maze import HEIGHT, WIDTH, Maze, States, Point

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import os

from copy import copy

class Node:
    def __init__(self, parent, cost):
        self.parent = parent
        self.cost = cost

class PriorityQueue:
    def __init__(self):
        pass

    def put(self, list, node, index):
        list[index] = node

    def get(self, openlist):
        lowest_f = inf
        for index in openlist:
            if openlist[index].cost < lowest_f:
                lowest_f = openlist[index].cost
                lowest_f_index = index
        return lowest_f_index

    def remove(self, openlist, index):
        del openlist[index]

class Graph:
    def __init__(self):
        pass

    def neighbours(self, pos, grid):
        x = pos.x
        y = pos.y
        neighbours = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)]

        neighbours = [pos for pos in neighbours if 0 <= pos[0] < WIDTH and 0 <= pos[1] < HEIGHT] # removes neighbours outside grid
        neighbours = [pos for pos in neighbours if grid[pos[1], pos[0]] != States.OBS.value] # removes any unoccupable space

        return neighbours

class Astar():
    def __init__(self):
        pass

    def heuristic(self, a, b):
        return abs(a.x - b.x) + abs(a.y - b.y)

    def f(self, g, h):
        return g+h
    
    def g(self, start, current):
        return self.heuristic(start, current)

    def h(self, current, goal):
        return self.heuristic(current, goal)
    
    def reconstruct_path(self, closedlist, start, goal, goalfound):
        path = []
        if goalfound:
            current = goal
            while closedlist[current].parent != -1: # note: this will fail if no path found
                path.append(current)
                current = closedlist[current].parent
            path.append(start) # optional
            path.reverse() # optional
            return path
        else:
            return path

    def run_astar(self, mode):
        PATH = os.getcwd()
        PATH = os.path.join(PATH, 'Results')
        PATH = os.path.join(PATH, 'Dijkstra')
        date_and_time = datetime.now()
        save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
        if not os.path.exists(save_path): os.makedirs(save_path)

        env = Maze()
        if mode != 1:
            PATH = os.getcwd()
            PATH = os.path.join(PATH, 'Results')
            load_path = os.path.join(PATH, 'Dijkstra')
            env.grid = extract_values(load_path, env)
        graph = Graph()

        print("Grid:\n", env.grid)
        print("Starting position: ", env.starting_pos)
        print("Goal: ", env.exit)

        # consists on nodes that have been visited but not expanded (meaning that sucessors have not been explored yet)
        queue = PriorityQueue()
        node = Node(parent=-1, cost=0)

        # consists on nodes that have been visited and expanded (sucessors have been explored already and included in the open list, if this was the case)
        closedlist = dict()
        openlist = dict()

        openlist[env.starting_pos] = copy(node)

        goalfound = False

        while openlist:
            current_pos = queue.get(openlist)
            current = openlist[current_pos]

            if current_pos == env.exit:
                goalfound = True
                closedlist[current_pos] = current
                break

            queue.remove(openlist, current_pos)
            closedlist[current_pos] = current
            
            for successor_pos in graph.neighbours(current_pos, env.grid):
                successor_pos = Point(successor_pos[0], successor_pos[1])
                successor_f = self.f(self.g(env.starting_pos, successor_pos), self.h(successor_pos, env.exit))

                if successor_pos in closedlist.keys(): continue
                
                if successor_pos not in openlist.keys():
                    node.parent = current_pos
                    node.cost = successor_f
                    openlist[successor_pos] = copy(node)
                else:
                    if self.g(env.starting_pos, successor_pos) < successor_f:
                        node.parent = current_pos
                        node.cost = successor_f
                        openlist[successor_pos] = copy(node)
                        pass
        
        path_taken = self.reconstruct_path(closedlist, env.starting_pos, env.exit, goalfound)
        if not path_taken:
            print("Path could not be found")
        else:
            env.grid[env.starting_pos.y, env.starting_pos.x] = States.ROBOT.value
            env.grid[env.exit.y, env.exit.x] = States.EXIT.value
            for i in np.arange(len(path_taken)):
                if i != 0:
                    env.grid[path_taken[i-1].y, path_taken[i-1].x] = States.EXP.value
                    env.grid[path_taken[i].y, path_taken[i].x] = States.ROBOT.value

                PR = print_results(env.grid, env.grid.shape[0], env.grid.shape[1])
                PR.print_graph(i)
                
                file_name = "plot-%s.png" %(i)
                plt.savefig(os.path.join(save_path, file_name))
                plt.close()
            
            f = open(os.path.join(save_path,"saved_data.txt"), "w", encoding="utf-8")
            f.write(str("Starting position: %s\nExit: %s" %(str(env.starting_pos), str(env.exit))))  
            f.close()
            file_name = "maze.txt"
            np.savetxt(os.path.join(save_path, file_name), env.grid)
        return 0

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

        plt_title = "A* Algorithm Results: Step %s" %(str(step))
        plt.title(plt_title)

def extract_values(load_path, env):
    f = open(os.path.join(load_path,"saved_data.txt"), "r")
    lines = f.readlines()
    WIDTH = 0
    HEIGHT = 0

    for line in lines:
        cur_num = ''
        cur_line = []
        if line[0:18] == "Starting position:":
            for num in line:
                if num.isdigit():
                    cur_num += num
                else:
                    if cur_num != '': cur_line.append(int(cur_num))
                    cur_num = ''
            env.starting_pos = Point(int(cur_line[0]),int(cur_line[1]))

        cur_num = ''
        cur_line = []
        if line[0:5] == "Exit:":
            for num in line:
                if num.isdigit():
                    cur_num += num
                else:
                    if cur_num == ',':
                        cur_line = []
                    elif cur_num != '': cur_line.append(int(cur_num))
                    cur_num = ''
            env.exit = Point(int(cur_line[0]),int(cur_line[1]))

    file_name = "maze.txt"
    return np.loadtxt(os.path.join(load_path, file_name))