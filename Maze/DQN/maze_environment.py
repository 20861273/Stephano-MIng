import numpy as np
import gym

class Environment():
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.grid = self.generate_grid()
        self.start_pos, self.exit_pos = self.generate_positions()

    def generate_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if np.random.random() < 0.2:
                    grid[i][j] = 1
        return grid

    def generate_positions(self):
        positions = np.random.choice(self.grid_size, size=4, replace=False)
        start_pos = (positions[0], positions[1])
        exit_pos = (positions[2], positions[3])
        return start_pos, exit_pos

    def reset(self):
        self.start_pos, self.exit_pos = self.generate_positions()
        self.current_pos = self.start_pos
        state = self.get_state()
        return state

    def _move(self, action):
        i, j = self.current_pos
        if action == 0: # move up
            i = max(0, i - 1)
        elif action == 1: # move down
            i = min(self.grid_size - 1, i + 1)
        elif action == 2: # move left
            j = max(0, j - 1)
        elif action == 3: # move right
            j = min(self.grid_size - 1, j + 1)
        self.current_pos = (i, j)

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.current_pos
        i, j = pt
        if self.grid[i][j] == 1:
            return True
        else:
            return False
    
    def _update_env(self):
        if self.current_pos == self.exit_pos:
            reward = 1
            done = True
        elif self._is_collision():
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        next_state = self.get_state()
        return next_state, reward, done

    def step(self, action):
        self._move(action)
        next_state, reward, done = self._update_env()
        return next_state, reward, done

    def get_state(self):
        state = self.grid.copy()
        state[self.current_pos] = 0.5 # mark current position
        state[self.exit_pos] = 0.7 # mark exit position
        return state