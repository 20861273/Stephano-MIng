from math import inf
from maze import HEIGHT, WIDTH, Maze, States, Point

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import os

class PriorityQueue:
    def __init__(self, openlist_shape, start, f):
        self.openlist = []
        self.openlist_costs = np.zeros(openlist_shape)
        self.put(start, f)

    def put(self, pos, f):
        self.openlist.append(pos)
        self.openlist_costs[pos[1], pos[0]] = f

    def get(self):
        lowest_f = inf
        index = 0
        for pos in self.openlist:
            if self.openlist_costs[pos[1], pos[0]] < lowest_f:
                lowest_f = self.openlist_costs[pos[1], pos[0]]
                lowest_f_pos = pos
            index += 1
        return lowest_f_pos, index

    def remove(self, element):
        self.openlist.remove(element)

class Graph:
    def __init__(self):
        pass

    def neighbours(self, pos, grid):
        x = pos[0]
        y = pos[1]
        neighbours = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)]

        neighbours = [pos for pos in neighbours if 0 <= x < WIDTH and 0 <= y < HEIGHT] # removes neighbours outside grid
        neighbours = [pos for pos in neighbours if grid[pos[1], pos[0]] != States.OBS.value] # removes any unoccupable space

        return neighbours

class Astar():
    def __init__(self):
        pass

    def heuristic(self, a, b):
        x1, x2 = a[0], b[0]
        y1, y2 = a[1], b[1]
        return abs(x1 - x2) + abs(y1 - y2)

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
            while current != start: # note: this will fail if no path found
                path.append(current)
                current = closedlist[current]
            path.append(start) # optional
            path.reverse() # optional
            return path
        else:
            return path

    def run_astar(self):
        env = Maze()
        graph = Graph()

        # consists on nodes that have been visited but not expanded (meaning that sucessors have not been explored yet)
        queue = PriorityQueue(env.grid.shape, env.starting_pos, 0)

        # consists on nodes that have been visited and expanded (sucessors have been explored already and included in the open list, if this was the case)
        closedlist = {}
        closedlist_costs = {}
        closedlist[env.starting_pos] = None
        closedlist_costs[env.starting_pos] = 0

        goalfound = False

        while queue.openlist:
            current, current_index = queue.get()

            if current == env.exit:
                goalfound = True
                break

            queue.remove(current)
            
            for successor in graph.neighbours(current, env.grid):
                successor_f = self.f(self.g(env.starting_pos, successor), self.h(successor, env.exit))
                queue.openlist_costs[successor[1], successor[0]] = successor_f
                if self.g(env.starting_pos, successor) < queue.openlist_costs[successor[1], successor[0]]:
                    closedlist_costs[next] = successor_f
                    closedlist[successor] = current
                    queue.put(successor, successor_f)
                    if successor not in queue.openlist: queue.put(successor, successor_f)
        
        path = self.reconstruct_path(closedlist, env.starting_pos, env.exit, goalfound)
        if not path:
            print("Path could not be found")
        else:
            pass
        # cost_so_far = dict()
        # came_from[start] = None
        # cost_so_far[start] = 0

        # while not frontier.empty():
        # current = frontier.get()

        # if current == goal:
        #     break
        
        # for next in graph.neighbors(current):
        #     new_cost = cost_so_far[current] + graph.cost(current, next)
        #     if next not in cost_so_far or new_cost < cost_so_far[next]:
        #         cost_so_far[next] = new_cost
        #         priority = new_cost + heuristic(goal, next)
        #         frontier.put(next, priority)
        #         came_from[next] = current
        return 0