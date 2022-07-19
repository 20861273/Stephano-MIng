from turtle import shape
import numpy as np
import random

# Python code to find number of unique paths
# in a Matrix
 
 
def UniquePathHelper(i, j, goal, A, paths, dir=True):
    # boundary condition or constraints
    if (i == A.shape[0] or j == A.shape[1]):
        dir = False
        return 0
 
    if (A[i][j] == 1):
        return 0
 
    # base case
    if (i == goal[1] - 1 and j == goal[0] - 1):
        return 1
 
    if (paths[i][j] != -1):
        return paths[i][j]
    
    if dir:
        return UniquePathHelper(i + 1, j, goal, A, paths) + UniquePathHelper(i, j + 1, goal, A, paths)
    else:
        return UniquePathHelper(i - 1, j, goal, A, paths) + UniquePathHelper(i, j - 1, goal, A, paths)
 
def uniquePathsWithObstacles(A, start, goal): 
    # create a 2D-matrix and initializing
    # with value 0
 
    paths = [[-1 for i in range(A.shape[1])]for j in range(A.shape[0])]
 
    return UniquePathHelper(start[1], start[0], goal, A, paths)
 


# Driver code
A  = np.ones((20,20))

# Set start and goal positions
s_quadrant = random.randint(0,3)
g_quadrant = (s_quadrant+2) % 4
quadrants = np.array([  [[A.shape[1]/4*3-1,A.shape[1]-1],   [0,A.shape[0]/4-1]],
                        [[0,A.shape[1]/4-1],                [0,A.shape[0]/4-1]],
                        [[0,A.shape[1]/4-1],                [A.shape[0]/4*3-1,A.shape[0]-1]],
                        [[A.shape[1]/4*3-1,A.shape[1]-1],   [A.shape[0]/4*3-1,A.shape[0]-1]]], dtype=int)

start = (random.randint(quadrants[s_quadrant,0,0],quadrants[s_quadrant,0,1]),
         random.randint(quadrants[s_quadrant,1,0],quadrants[s_quadrant,1,1]))

goal = (random.randint(quadrants[g_quadrant,0,0],quadrants[g_quadrant,0,1]),
        random.randint(quadrants[g_quadrant,1,0],quadrants[g_quadrant,1,1]))

A[start[1], start[0]] = 3
A[goal[1], goal[0]] = 4

obs_per = 100
obs_per_lim = 40
paths = 0
print(start, goal)

while not paths > 2 or obs_per < obs_per_lim:
    
    paths = uniquePathsWithObstacles(A, start, goal)
    obs_indices = np.argwhere(A == 1)
    np.random.shuffle(obs_indices)
    obs_per = (obs_indices.shape[0])/np.size(A)*100
    print(A, paths, obs_per)
    per_switch = int(0.2*len(obs_indices))
    obs_remaining = np.size(A) - per_switch
    path_indices, obs_indices = np.split(obs_indices, [per_switch])
    
    for index in path_indices:
        A[index[0], index[1]] = 0

