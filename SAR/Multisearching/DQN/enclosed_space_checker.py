import numpy as np

class Enclosed_space_check:
    def __init__(self, WIDTH, HIGHT, EnvironmentGrid, States):
        self.width = WIDTH
        self.height = HIGHT
        self.States = States
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
        
        index = max(range(len(labels)), key=lambda i: len(labels[i]))
        del labels[index]
        combined_labels = []
        for l in labels: combined_labels.extend(l)
        return combined_labels
    
    def enclosed_space_handler(self):
        for p in self.final_labels:
            self.binary_grid[p[1], p[0]] = self.States.OBS.value

        return self.binary_grid
