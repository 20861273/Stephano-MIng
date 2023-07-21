import numpy as np
import torch as T
import random

# last at sample buffer

class ReplayBuffer(object):
    def __init__(self, lidar, max_size, input_shape, n_actions):
        self.lidar = lidar # TODO
        self.mem_size = max_size
        self.mem_cntr = 0 # mem_cntr of the last stored memory
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        # self.non_image_state_memory = np.zeros((self.mem_size, 2),
        #                              dtype=np.float32)
        self.non_image_state_memory = np.zeros((self.mem_size, 1),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        # self.new_non_image_state_memory = np.zeros((self.mem_size, 2),
        #                              dtype=np.float32)
        self.new_non_image_state_memory = np.zeros((self.mem_size, 1),
                                     dtype=np.float32)
        

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done, non_image_state, non_image_state_):#image_state, action, reward, image_state_, done, non_image_state, non_image_state_
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.non_image_state_memory[index] = non_image_state
        self.new_non_image_state_memory[index] = non_image_state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        non_image_states = self.non_image_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        new_non_image_state_memory = self.new_non_image_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, non_image_states, actions, rewards, states_, new_non_image_state_memory, terminal 

class PrioritizedReplayMemory(object):
    def __init__(self, lidar, obs_shape, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.lidar = lidar

        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]

        self.max_priority = 1.

        if self.lidar:
            self.data = {
                'obs': np.zeros(shape=((capacity,) + tuple(obs_shape)), dtype=np.float32),
                'lidar_obs': np.zeros(shape=((capacity,) + tuple(obs_shape)), dtype=np.float32),
                'action': np.zeros(shape=capacity, dtype=np.int32),
                'reward': np.zeros(shape=capacity, dtype=np.float32),
                'next_obs': np.zeros(shape=((capacity,) + tuple(obs_shape)), dtype=np.float32),
                'next_lidar_obs': np.zeros(shape=((capacity,) + tuple(obs_shape)), dtype=np.float32),
                'done': np.zeros(shape=capacity, dtype=np.bool_)
            }
        else:
            self.data = {
                'obs': np.zeros(shape=((capacity,) + tuple(obs_shape)), dtype=np.float32),
                'action': np.zeros(shape=capacity, dtype=np.int32),
                'reward': np.zeros(shape=capacity, dtype=np.float32),
                'next_obs': np.zeros(shape=((capacity,) + tuple(obs_shape)), dtype=np.float32),
                'done': np.zeros(shape=capacity, dtype=np.bool_)
            }

        self.next_idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done, lidar_obs=None, next_lidar_obs=None):
        idx = self.next_idx

        self.data['obs'][idx] = obs
        self.data['action'][idx] = action
        self.data['reward'][idx] = reward
        self.data['next_obs'][idx] = next_obs
        self.data['done'][idx] = done

        if self.lidar:
            self.data['lidar_obs'][idx] = lidar_obs
            self.data['next_lidar_obs'][idx] = next_lidar_obs

        self.next_idx = (idx + 1) % self.capacity

        self.size = min(self.capacity, self.size + 1)

        priority_alpha = self.max_priority ** self.alpha

        self._set_priority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx //= 2

            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.capacity

        self.priority_sum[idx] = priority

        while idx >= 2:

            idx //= 2

            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
    
    def _sum(self):
        return self.priority_sum[1]
    
    def _min(self):
        return self.priority_min[1]
    
    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1

        return idx - self.capacity

    def sample(self, batch_size, beta):
        samples = {
            'weights': np.zeros(shape=batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=batch_size, dtype=np.int32)
        }

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx
        
        prob_min = self._min() / self._sum()

        max_weight = (prob_min * self.size) ** (-beta)

        for i in range(batch_size):
            idx = samples['indexes'][i]

            prob = self.priority_sum[idx + self.capacity] / self._sum()

            weight = (prob * self.size) ** (-beta)

            samples['weights'][i] = weight / max_weight

        for k, v in self.data.items():
            samples[k] = v[samples['indexes']]

        return samples

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)

            priority_alpha = priority ** self.alpha

            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_full(self):
        return self.capacity == self.size
