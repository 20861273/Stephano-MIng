from collections import namedtuple
import numpy as np

class Node:
    def __init__(self, cost, parent_index):
        self.cost = cost
        self.parent_index = parent_index

Point = namedtuple('Point', 'x, y')

node = Node(0,-1)
start = Point(0,0)
end = Point(5,5)

hi = dict()
hi[start] = node
hi[end] = node

g = np.zeros((10,10))
c = [Point(0,0), Point(10,1)]

c = [pos for pos in c if 0 <= pos.x < 10 and 0 <= pos.y < 10]
print(c)
