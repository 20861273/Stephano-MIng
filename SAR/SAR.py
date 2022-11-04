from re import T
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

# Global variables

# Plotting varibales
LINEWIDTH = 0.7
S_MARKERIZE = LINEWIDTH*4
MARKERSIZE=LINEWIDTH*12

# Setting path to project
PATH = os.getcwd()
PATH = os.path.join(PATH, 'Results')
date_and_time = datetime.now()
PATH = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
try:
    os.mkdir(PATH)
except OSError:
    try:
        os.mkdir(PATH, exist_ok=True)
    except OSError:
        #Folder already exsits
        pass

# Navigation
VEL = 1 # space / s
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

# Time
EP_TIME = 0
TIMEOUT = 10

# Position state
UNEXP = 0
OBS = 1
ROBOT = 2
TARGET = 3
EXP = 4

class robots:
    def __init__(self,n):
        self.n = n
        self.rip = np.zeros((n,2),dtype=int)
        self.pos = np.zeros((1,n,2),dtype=int)
        self.dir = np.zeros(n,dtype=int)

    def randomise_positions(self,world):
        """
        Constructs all the necessary attributes for the robots' object.

        Parameters
        ----------
            grid : int
                3D array of grid-based environment at each time step. (grid[time_step, y, x])
        """

        # Set initial positions
        possible_indexes = np.argwhere(world.grid[0] == 0)
        np.random.shuffle(possible_indexes)

        # Checks if there are too many robots for environment size
        if self.n < world.grid.shape[0]*world.grid.shape[1]:
            self.pos[0] = possible_indexes[0:self.n]
            possible_indexes = np.delete(possible_indexes,np.arange(0,self.n,1),0)
            self.rip = self.pos[0]
        else:
            print("Bruh.. you got too many robots.")
        
class target:
    def __init__(self):
        self.pos = np.zeros((1,2),dtype=int)
    
    def randomise_position(self,world):
        """
        Constructs all the necessary attributes for the target's object.

        """

        # Set initial positions
        possible_indexes = np.argwhere(world.grid[0] == 0)
        np.random.shuffle(possible_indexes)
        self.pos = possible_indexes[0:1]

class environment:
    """
    A class used to generate the grid-based environment

    ...

    Attributes
    ----------
    rows : int
        number of rows in the environment
    cols : int
        number of columns in the environment
    grid : int
        3D array of grid-based environment at each time step. (grid[time_step, y, x])
    n_r : int
        the number of robots
    rip : int
        robot inital position
    r_dir : int
        robot direction
    target : int
        target position
    Methods
    -------
    randomise_robots(self,n_r):
        sets robots' initial positions and directions
    def setTarget(self,grid):
        sets target's initial position
    """

    def __init__(self,hor,vert):
        """
        Constructs all the necessary attributes for the grid object.

        Parameters
        ----------
            hor : int
                horizontal length of environment
            vert : int
                vertical length of environment
        """
        self.rows = vert
        self.cols = hor
        self.grid = np.zeros([1,self.rows, self.cols], dtype=int)
    
    def update_grid(self,step,ry,rx,target):
        # Update robot position(s) on grid
        self.grid[step,ry,rx] = ROBOT

        # Update target position on grid
        tx = target.pos[:, 0]
        ty = target.pos[:, 1]
        self.grid[step,tx,ty] = TARGET

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

    def __init__(self,grid,rows,cols,n_r):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.n_r = n_r
    def print_graph(self):
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
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'b', alpha=0.75)
                elif self.grid[j][i] == 2:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'g', alpha=0.75)
                elif self.grid[j][i] == 3:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'r', alpha=0.75)
                elif self.grid[j][i] == 4:
                    plt.fill( [x1, x1, x2, x2], [y1, y2, y2, y1], 'c', alpha=0.75)
                    
        plt.title("Results")             

class LMP:
    """
    A class used to plan the lawnmower pattern

    Methods
    -------
    def lawnmower_main(self,r_dir):
        plans and executes lawn mower pattern
    """
    def __init__(self,total_itr):
        self.steps = np.zeros((total_itr,1),dtype=int)
    def lawnmower_main(self,world,robot,target,itr):
        """
        Plans and executes lawn mower pattern

        Parameters
        ----------
            r_dir : int
                direction of robots
        """
        sweep = np.ones(robot.n)
        robot.dir = np.full(np.shape(robot.dir), RIGHT)
        prev_dir = list(robot.dir)
        x = np.zeros((robot.n,2),dtype=int)
        y = np.zeros((robot.n,2),dtype=int)

        self.steps[itr,0] = robot.pos.shape[0]-1
        
        for i in np.arange(robot.n):
            robot.pos[robot.pos.shape[0]-1,i] = robot.rip[i]
        
        while True:
            #print("\nStep: ", robot.pos.shape[0]-1)            
            y = np.array(robot.pos[self.steps[itr,0],:,0],dtype=int)
            x = np.array(robot.pos[self.steps[itr,0],:,1],dtype=int)

            cur_grid = np.array(world.grid[self.steps[itr,0]], dtype=int)

            cur_grid[y, x] = EXP # Updates current blocks

            # Checks for boudaries and updates robots direction if necessary
            for i in np.arange(robot.n):
                if sweep[i] == UP: # Up sweep
                    if robot.dir[i] == UP:
                        if prev_dir[i] == RIGHT:
                            prev_dir[i] = robot.dir[i]
                            robot.dir[i] = LEFT
                        elif prev_dir[i] == LEFT:
                            prev_dir[i] = robot.dir[i]
                            robot.dir[i] = RIGHT
                    elif (x[i] >= world.cols-1 and robot.dir[i] == RIGHT) or (x[i] <= 0 and robot.dir[i] == LEFT): # Vertical wall collision checks
                        if cur_grid[0, 0] == 4 and cur_grid[0, world.rows-1] == 4:
                            print("Target not found!!!")
                            return
                        if y[i] <= 0:
                            if cur_grid[0, 0] == UNEXP: 
                                prev_dir[i] = robot.dir[i]
                                robot.dir[i] = LEFT
                            elif cur_grid[0, world.cols-1] == UNEXP: 
                                prev_dir[i] = robot.dir[i]
                                robot.dir[i] = RIGHT
                        else:
                            prev_dir[i] = robot.dir[i]
                            robot.dir[i] = UP
                if sweep[i] == DOWN: # Down sweep
                    if robot.dir[i] == DOWN:
                        if prev_dir[i] == RIGHT:
                            prev_dir[i] = robot.dir[i]
                            robot.dir[i] = LEFT
                        elif prev_dir[i] == LEFT:
                            prev_dir[i] = robot.dir[i]
                            robot.dir[i] = RIGHT
                    elif (x[i] >= world.cols-1 and robot.dir[i] == RIGHT) or (x[i] <= 0 and robot.dir[i] == LEFT):
                        if y[i] >= world.rows-1:
                            if cur_grid[world.rows-1, 0] == UNEXP: 
                                prev_dir[i] = robot.dir[i]
                                robot.dir[i] = LEFT
                            elif cur_grid[world.rows-1, world.cols-1] == UNEXP: 
                                prev_dir[i] = robot.dir[i]
                                robot.dir[i] = RIGHT
                            elif cur_grid[world.rows-1, 0] == EXP and cur_grid[world.rows-1, world.cols-1] == EXP:
                                prev_dir[i] = robot.dir[i]
                                robot.dir[i] = UP
                                sweep[i] = UP
                        
                        else:
                            prev_dir[i] = robot.dir[i]
                            robot.dir[i] = DOWN
                
                # Debugging
                # if robot.dir[i] == UP: print(x,y,"UP",cur_grid[world.rows-1, 0],cur_grid[world.rows-1, world.cols-1])
                # if robot.dir[i] == DOWN: print(x,y,"DOWN",cur_grid[world.rows-1, 0],cur_grid[world.rows-1, world.cols-1])
                # if robot.dir[i] == RIGHT: print(x,y,"RIGHT",cur_grid[world.rows-1, 0],cur_grid[world.rows-1, world.cols-1])
                # if robot.dir[i] == LEFT: print(x,y,"LEFT",cur_grid[world.rows-1, 0],cur_grid[world.rows-1, world.cols-1])
                # f = open(os.path.join(PATH,"output.txt"), "w")
                # f.write(str(cur_grid))
                # f.close()

                # Updates robots x or y positions
                if robot.dir[i] == UP:
                    y[i] = y[i]-1
                elif robot.dir[i] == DOWN:
                    y[i] = y[i]+1
                elif robot.dir[i] == RIGHT:
                    x[i] = x[i]+1
                elif robot.dir[i] == LEFT:
                    x[i] = x[i]-1

                cur_grid[int(y[i]), int(x[i])] = EXP # Updates current blocks

            # Add robot positions of current time step
            cur_r_pos = np.zeros((robot.n,2))
            cur_r_pos[:,0] = y
            cur_r_pos[:,1] = x
            tmp_r_pos = np.array(robot.pos)
            new_pos = np.array(np.append(tmp_r_pos.ravel(),cur_r_pos.ravel()))
            robot.pos = new_pos.reshape(robot.pos.shape[0]+1,robot.pos.shape[1],robot.pos.shape[2])

            self.steps[itr,0] = robot.pos.shape[0]-1

            # Add grid of current time step
            tmp_grid = np.array(world.grid)
            new_grid = np.array(np.append(tmp_grid.ravel(),cur_grid.ravel()))
            world.grid = new_grid.reshape(world.grid.shape[0]+1,world.grid.shape[1],world.grid.shape[2])

            world.update_grid(self.steps[itr,0],y,x,target)

            # Checks of target has been found
            for i in range(robot.pos.shape[1]):
                if np.all(robot.pos[self.steps[itr,0],i]==target.pos):
                    print("Target found!!!")
                    return 

            

if __name__ == "__main__":
    # Environment size
    horizontal = 10
    vertical = 10

    # Number of robots
    n_r = 2

    # Number of iterations
    total_itr = 10

    # Boolean variables
    print_graphs = False
    save_data = True
    save_plots = False

    for itr in np.arange(total_itr):
        print("\nIteration: ", itr)
        # Generates grid
        GG = environment(horizontal,vertical)

        # Setup robot object
        R = robots(n_r)
        R.randomise_positions(GG)

        # Setup target object
        T = target()
        T.randomise_position(GG)
        GG.update_grid(0,np.array(R.pos[0,:,0],dtype=int),np.array(R.pos[0,:,1],dtype=int),T)
        
        start_str = str("Grid size: "+ str(GG.rows)+ " x "+ str(GG.cols)+"\nNumber of robots: "+ str(R.n) +"\nRobot initial positions:\n"+ str(R.pos)+"\nTarget position: "+ str(T.pos))
        print(start_str)

        # Select and run algorithm
        # Algorithms:
        # 0     -       Lawnmower
        # 1     -       DARP
        # 2     -       RL
        # 3     -       A*
        sel_algorithm = 0

        start_time = time.time()
        LP = LMP(total_itr)
        LP.lawnmower_main(GG,R,T,itr)
            
        #elif sel_algorithm == 1:
        #    DARP()
        #elif sel_algorithm == 2:
        #    RL()
        #elif sel_algorithm == 2:
        #    Astar()

        EP_TIME = time.time() - start_time
        print("Total steps: ", LP.steps[itr,0])
        print("Total time: ", EP_TIME, " s")
        
        # give each attribute one for dimension for itr
        # Saves results
        if save_data:
            path = os.getcwd()
            path = PATH
            path = os.path.join(path, str(itr))
            try:
                os.mkdir(path)
            except OSError:
                try:
                    os.mkdir(path, True)
                except OSError:
                    #print("Folder " + str(folder) + " already exists")
                    pass

            f = open(os.path.join(path,"output.txt"), "w")
            if itr == 0: f.write(str("Iterations: "+str(total_itr)+"\nNumber of robots: "+str(n_r)))
            f.write(str("Interation: "+str(itr)+"\nRobots position: (y,x)\n"+str(R.pos)+"\nTarget position: (y,x)\n"+str(T.pos)))
            f.close()

            if save_plots:
                for i in np.arange(GG.grid.shape[0]):
                    for j in np.arange(n_r):
                        GG.grid[i, int(R.pos[i,j,0]), int(R.pos[i,j,1])] = ROBOT
                    PR = print_results(GG.grid[i], GG.rows, GG.cols, n_r)
                    PR.print_graph()
                    file_name = 'plot'+str(i)+'.png'
                    plt.savefig(os.path.join(path, file_name))
                    plt.close()

        # Prints results
        if print_graphs:
            PR = print_results(LP.grid[i], GG.rows, GG.cols, n_r, LP.r_pos)
            PR.print_graph()
            plt.show()

"""
    TO DO:
    - Setup robot class
    - Setup target class
    - Setup grid update function
"""