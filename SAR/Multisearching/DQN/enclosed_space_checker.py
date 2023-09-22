import numpy as np
from enum import Enum

class Enclosed_space_check:
    def __init__(self, WIDTH, HIGHT, EnvironmentGrid, States):
        self.width = WIDTH
        self.height = HIGHT
        self.States = States
        self.spaces = []
        self.binary_grid = np.copy(EnvironmentGrid)
        self.final_labels = self.labeling()


    def labeling(self):
        cnter = 0
        labels = []

        for x in range(self.width):
            for y in range(self.height):
                if self.binary_grid[y,x] != self.States.UNEXP.value:
                    continue
                if cnter == 0:
                    labels.append([(x,y)])
                    cnter += 1
                
                # checks if (x,y) is in region
                # if not then add to new region
                added = False
                for i in range(len(labels)):
                    if (x,y) in labels[i]:
                        added = True
                        index = i
                if not added:
                    labels.append([(x,y)])
                    index = len(labels)-1

                # checks if right is connected
                if x != self.width-1:
                    if self.binary_grid[y,x] == self.binary_grid[y,x+1]:
                        # checks if (x+1,y) is in a label list already
                        combined = False
                        for i, l in enumerate(labels):
                            if (x+1,y) in labels[i] and i != index:
                                combine_lists = labels[i] + labels[index]
                                labels[min(i, index)] = combine_lists
                                del labels[max(i, index)]
                                combined = True
                        if not combined: 
                            labels[index].append((x+1,y))

                # checks if bottom is connected
                if y != self.height-1:
                    if self.binary_grid[y,x] == self.binary_grid[y+1,x]:
                        # checks if (x+1,y) is in a label list already
                        combined = False
                        for i, l in enumerate(labels):
                            if (x,y+1) in labels[i] and i != index:
                                combine_lists = labels[i] + labels[index]
                                labels[min(i, index)] = combine_lists
                                del labels[max(i, index)]
                                combined = True
                        if not combined:
                            labels[index].append((x,y+1))
        
        if len(labels) == 1:
            return []
        
        # save largest spaces
        # index = max(range(len(labels)), key=lambda i: len(labels[i]))
        # self.spaces = []
        # labels.extend(index)
        # del labels[index]
        save_index = []
        for i,l in enumerate(labels):
            if len(l) > self.height*self.width*0.1:
                save_index.append(i)
        self.spaces = [labels[save_index[0]]]
        if len(save_index) != 1:
            for i in range(1, len(save_index)):
                self.spaces.append(labels[save_index[i]])
        # delete saved indices
        labels = [labels[i] for i in range(len(labels)) if i not in save_index]

        combined_labels = []
        for l in labels: combined_labels.extend(l)
        return combined_labels

    def combine_spaces(self):
        distances = {}
        
        # select position in space
        for i,p1 in enumerate(self.spaces[0]):
            shortest_distance = self.height*self.width
            # select comparing space
            for l2 in range(1, len(self.spaces)):
                # select position in comparing space
                for j,p2 in enumerate(self.spaces[l2]):
                    distance = self.get_distance(p1,p2)
                    if distance < shortest_distance:
                        shortest_distance = distance
                        distances[l2] = [p1,p2]
        
        for i,l in enumerate(distances):
            x_distance = abs(distances[l][0][0] - distances[l][1][0])
            y_distance = abs(distances[l][0][1] - distances[l][1][1])
            x1 = distances[l][0][0]
            y1 = distances[l][0][1]
            x2 = distances[l][1][0]
            y2 = distances[l][1][1]
            step = 0
            condition = True
            while condition:
                if step % 2 == 0: # x direction
                    if x1 < x2:
                        x1 += 1
                    else:
                        x1-= 1
                    if x_distance != 0: x_distance -= 1
                else: # y direction
                    if y1 < y2:
                        y1 += 1
                    else:
                        y1-= 1
                    if y_distance != 0: y_distance -= 1
                self.binary_grid[y1, x1] = self.States.UNEXP.value
                step += 1
                if x_distance == 0 and y_distance == 0:
                    condition = False
        breakpoint
    
    def get_distance(self, start, end):
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def enclosed_space_handler(self):
        for p in self.final_labels:
            self.binary_grid[p[1], p[0]] = self.States.OBS.value

        if len(self.spaces) > 1:
            # print("\n\n",self.binary_grid)
            self.combine_spaces()
            # print("\n",self.binary_grid)

        return self.binary_grid
    

# class States(Enum):
#     UNEXP = 0
#     OBS = 1
#     ROBOT = 2
#     GOAL = 3
#     EXP = 4

# h,w=6,6

# for i in range(100):
#     grid = np.zeros((h, w))

#     # obstacles
#     starting_grid = grid.copy()

#         # Calculate the number of elements to be filled with 1's
#     total_elements = int(h) * int(w)
#     num_ones_to_place = int(0.3 * total_elements)

#         # Generate random indices to place 1's
#     possible_indexes = np.argwhere(np.array(grid) == 0)
#     np.random.shuffle(possible_indexes)
#     indexes = possible_indexes[:num_ones_to_place]

#         # Set the elements at the random indices to 1
#     starting_grid[indexes[:, 0], indexes[:, 1]] = 1

#     # print(starting_grid)

#     ES = Enclosed_space_check(h, w, starting_grid, States)
#     grid = ES.enclosed_space_handler()

#     if len(np.argwhere(np.array(grid) == 1).tolist()) > 10:
#         print("%d\n"%(i))
#         print(starting_grid)
#         print("\n", grid)
