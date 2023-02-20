import heapq
from collections import namedtuple
import collections

from astar_environment import Environment, HEIGHT, WIDTH, Point

class GridWithWeights(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.weights = {}
        self.visited_blocks = set()

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.weights

    def cost(self, current, next):
        if next in self.visited_blocks:
            return self.weights.get(current, 10)
        else:
            return self.weights.get(current, 1)

    def neighbors(self, id):
        (x, y) = id.x, id.y
        # (right, up, left, down)
        results = [Point(x+1, y), Point(x, y-1), Point(x-1, y), Point(x, y+1)]
        # This is done to prioritise straight paths
        #if (x + y) % 2 == 0: results.reverse()
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

class Queue:
    def __init__(self):
        self.elements = collections.deque()
    
    def empty(self):
        return not self.elements
    
    def put(self, x):
        self.elements.append(x)
    
    def get(self):
        return self.elements.popleft()

class PriorityQueue:
    def __init__(self):
        self.elements: list[tuple[float]] = []
    
    def empty(self):
        return not self.elements
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

# Heuristic function to estimate the cost from a current block to a end block
def heuristic(current, end):
    return abs(current[0] - end.x) + abs(current[1] - end.y)

def reconstruct_path(came_from, start, goal): #: dict[Location, Location], : Location, : Location
    current = end
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# A* algorithm
def a_star(graph, start, end):
    heap = [(0, start)]
    came_from = {}
    cost_so_far = {}
    cost_so_far[start] = 0
    
    while heap:
        current = heapq.heappop(heap)[1]
        print(current)
        if current == end:
            break
        
        nn = []
        for next_node in graph.neighbors(current):
            nn.append(next_node)
            new_cost = cost_so_far[current] + graph.cost(current, next_node)
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(end, next_node)
                heapq.heappush(heap, (priority, next_node))
                came_from[next_node] = current
        print(nn)
                
    return came_from, cost_so_far

env = Environment()
end = env.goal

graph = GridWithWeights(WIDTH, HEIGHT)

came_from, cost_so_far = a_star(graph, env.starting_pos, end)
path = reconstruct_path(came_from, env.starting_pos, end)
print("Path:", path)
print("Cost:", cost_so_far[end])
print("Grid: \n", env.grid)