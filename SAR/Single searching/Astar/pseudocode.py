from operator import itemgetter

class States:
    def __init__(self, position=(0,0), fuel=0):
        self.position = position
        self.fuel = fuel
        self.score_to_come = 0
        self.score_to_go = 0
        self.total_score = 0
        self.ID = 0
        self.parent_ID = -1
    
class PriorityQueue:
    def __init__(self):
        self.elements = []
    def empty(self):
        return not self.elements
    def insert(self, item):
        self.elements.append(item)
    def getFirst(self):
        return self.elements[0]
    def sort(self):
        self.elements.sort(key=lambda x: x.score_to_come)

initial_state = States()

initial_state.position      = (xi,yi)

initial_state.fuel          = N

initial_state.score_to_come = 0

closest_unexplored_cell = States()

if initial_state.fuel >= distance(initial_state.position, closest_unexplored_cell.position) + distance(closest_unexplored_cell.position, initial_state.position):

    initial_state.score_to_go = 1

else:

    # no unexplored cells within range of initial position, return empty path

    return empty

initial_state.position      = (xi,yi)

initial_state.fuel          = N

initial_state.score_to_come = 0

if initial_state.fuel >= distance(initial_state.position, closest_unexplored_cell.position) + distance(closest_unexplored_cell.position, initial_state.position):

    initial_state.score_to_go = 1

else:

    # no unexplored cells within range of initial position, return empty path

    return empty

initial_state.total_score   = initial_state.score_to_come + initial_state.score_to_go

initial_state.ID            = 0

initial_state.parent_ID     = -1

tree[0]                     = initial_state

PriorityQueue.insert(initial_state)

state_ID                    = 0

while PriorityQueue not empty:

    current_state = PriorityQueue.getFirst()

    # if used all fuel and drone is back at initial position, then goal reached (allow a margin of 1 fuel for even/odd number of moves)

    if current_state.fuel <= 1 and current_state.position == initial_state.position:

        # reconstruct best coverage path by starting at last state and iterating backwards through parents states

        path_list = [current_state]

        while current_state.parentID != -1:

            current_state = state_list[current_state.parentID]

            path_list.insert(0, current_state)

        return path_list

    # otherwise, generate child states

    for action in action_set:

        next_state.position = move(current_state.position, action) # calculate next position

        next_state.fuel     = current_state.fuel - 1               # reduce fuel by one

        # check whether there is enough fuel left to get back to the start position from the next state

        # if the initial position cannot be reached from the next state, then do not add the next state to the tree or the priority queue

        # if the initial condition can be reached from the next state, then calculate the score of the next state and add it to the tree and the priority queue

        if next_state.fuel < distance(next_state.position, initial_state.position)

            # do not add next_state to tree or priority queue

        else:

            # the score to come is the score to come so far plus the unexplored status (1 or 0) of the next state

            next_state.score_to_come = current_state.score_to_come + unexplored_status[next_state.position]

            # the next state's score to go is 1 if there is any unexplored cell that the drone can reach and also return to the initial position afterwards

            # the next state's score to go is 0 if there are no unexplored cells within reach, but the drone can still reach the initial position

            if next_state.fuel >= distance(next_state.position, closest_unexplored_cell.position) + distance(closest_unexplored_cell.position, initial_state.position):

                next_state.score_to_go   = 1

            else if next_state.fuel >= distance(next_state.position, initial_state.position):

                next_state.score_to_go   = 0

            # the total score is the sum of the score to come and the score to go

            next_state.score = next_state.score_to_come + next_state.score_to_go

            # increment the state ID, record the parent state ID, and add the next state to the tree and the priority queue

            state_ID            = state_ID + 1

            next_state.ID       = state_ID

            next_state.parentID = current_state.ID

            tree[state_ID]      = next_state

            PriorityQueue.insert(next_state)

    # after generating all the child states from the current state, sort the priority queue from highest to lowest score

    PriorityQueue.sort()

# if the priority queue is empty and the initial node was never reached after using all the fuel, then return empty path

return empty