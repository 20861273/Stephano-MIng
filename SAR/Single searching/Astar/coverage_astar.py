from operator import itemgetter

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

    def cost(self, current, next_node, closed_set):
        branch = reconstruct_path(closed_set, current)
        if next_node in branch:
            return self.weights.get(current[4], 0)
        else:
            return self.weights.get(current[4], 1)

    def neighbors(self, id):
        (x, y) = id.x, id.y
        # (right, up, left, down)
        results = [Point(x, y), Point(x+1, y), Point(x, y-1), Point(x-1, y), Point(x, y+1)]
        # This is done to prioritise straight paths
        # if (x + y) % 2 == 0: results.reverse()
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

def reconstruct_path(closed_set, current):
    current = current[1:5]
    path = [current[3]]
    while current[0] != -1:
        current = closed_set[current[0]]
        path.append(current[3])
    path.reverse()
    return path

# def traverse_tree(closed_set):
#     t = closed_set[0][1]
#     while 
#     for i in closed_set:
#         if i[1] != t:
#             t = i[1]
#             print("Time ", t, ":")
#         print(i)


# A* algorithm
def a_star(graph, start, termination_reward):
    open_set = [(0, -1, 0, 0, start)] # id, parent_id, time, reward, position
    closed_set = {}
    closed_set[0] = (-1, 0, 0, start)
    id = 1
    t = 0
    
    
    while len(open_set) > 0:
        open_set = list(sorted(open_set, key=itemgetter(3, 0)))
        current = open_set[-1]
        open_set.pop()
        if current[4] == start and t != 0 and termination_reward == current[3]:
            break
        
        if current[3] <= termination_reward:
            for next_node in graph.neighbors(current[4]):
                new_cost = current[2] + graph.cost(current, next_node, closed_set)
                if next_node not in open_set or new_cost < current[3]:
                    t = current[2] + 1
                    open_set.append((id, current[0], t, new_cost, next_node))
                    closed_set[id] = (current[0], t, new_cost, next_node)
                id += 1
            
                
    return closed_set, open_set, current

env = Environment()

graph = GridWithWeights(WIDTH, HEIGHT)

env.starting_pos = Point(2,2)

termination_reward = 3

closed_set, open_set, last = a_star(graph, env.starting_pos, termination_reward)
path = reconstruct_path(closed_set, last)[:-1]
# print("Tree: \n", traverse_tree(closed_set))
print("Path:", path)
print("Cost:", closed_set[last[0]][2])
print("Grid: \n", env.grid)

i = 1
for p in path:
    env.grid[p.y, p.x] = i
    i += 1

print("Path:\n", env.grid)