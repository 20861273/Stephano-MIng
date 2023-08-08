import heapq
from collections import namedtuple
import collections

from astar_environment import Environment, HEIGHT, WIDTH, Point

class GridWithWeights(object):
    def __init__(self, height, width): ########################################### change
        self.width = width
        self.height = height

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, id):
        (x, y) = id
        # (right, up, left, down)
        results = [Point(x+1, y), Point(x, y-1), Point(x-1, y), Point(x, y+1)]
        # This is done to prioritise straight paths
        #if (x + y) % 2 == 0: results.reverse()
        results = list(filter(self.in_bounds, results))
        return results

class Astar:
    def __init__(self):
        self.came_from = {}
        self.cost_so_far = {}
        self.graph = GridWithWeights(HEIGHT, WIDTH)

    # Heuristic function to estimate the cost from a current block to a end block
    def heuristic(self, current, end):
        return abs(current[0] - end.x) + abs(current[1] - end.y)
    
    # Path reconstruction
    def reconstruct_path(self, start, end): #: dict[Location, Location], : Location, : Location
        current = end
        path = [current]
        while current != start:
            current = self.came_from[current]
            path.append(current)
        path.reverse()
        return path

    # A* algorithm
    def a_star(self, start, end):
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
                if next_node not in self.cost_so_far or new_cost < self.cost_so_far[next_node]:
                    self.cost_so_far[next_node] = new_cost
                    heapq.heappush(heap, (new_cost, next_node))
                    self.came_from[next_node] = current
        
        return self.reconstruct_path(start, end)

env = Environment()
end = env.goal

astar = Astar()

path = astar.a_star(env.starting_pos, end)

print("Path:", path)
print("Cost:", astar.cost_so_far[end])
print("Grid: \n", env.grid)