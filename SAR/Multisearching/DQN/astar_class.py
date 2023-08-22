import heapq
from collections import namedtuple
import collections

Point = namedtuple('Point', 'x, y')

class GridWithWeights(object):
    def __init__(self, height, width, grid, States): ########################################### change
        self.grid = grid
        self.width = width
        self.height = height
        self.States = States

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def is_unoccupied(self, id):
        (x, y) = id
        return self.grid[y,x] != self.States.OBS.value

    def neighbors(self, id):
        (x, y) = id
        # (right, up, left, down)
        results = [Point(x+1, y), Point(x, y-1), Point(x-1, y), Point(x, y+1)]
        # This is done to prioritise straight paths
        #if (x + y) % 2 == 0: results.reverse()
        results = list(filter(self.in_bounds, results))
        results = list(filter(self.is_unoccupied, results))
        return results

class Astar:
    def __init__(self, height, width, grid, States):
        self.came_from = {}
        self.cost_so_far = {}
        self.grid = grid
        self.States = States
        self.graph = GridWithWeights(height, width, grid, States)

    # Heuristic function to estimate the cost from a current block to a end block
    def heuristic(self, current, end):
        return abs(current[0] - end.x) + abs(current[1] - end.y)
    
    # Path reconstruction
    def reconstruct_path(self, start, end): #: dict[Location, Location], : Location, : Location
        current = end
        path = [current]
        while current != start:
            if current not in self.came_from: print(self.grid, self.start, end)
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path

    # A* algorithm
    def a_star(self, start, end):
        self.start = start
        self.came_from = {}
        self.cost_so_far = {}
        heap = [(0, start)]
        self.cost_so_far[start] = 0
        current = start
        
        while heap:
            if current == end:
                break
            current = heapq.heappop(heap)[1]
            for next_node in self.graph.neighbors(current):
                new_cost = self.cost_so_far[current] + self.heuristic(current, next_node) + self.heuristic(next_node, end)
                if next_node not in self.cost_so_far or new_cost < self.cost_so_far[next_node]: #self.grid[next_node.y, next_node.x] != self.States.OBS.value
                    self.cost_so_far[next_node] = new_cost
                    heapq.heappush(heap, (new_cost, next_node))
                    self.came_from[next_node] = current
        
        return self.reconstruct_path(start, end)