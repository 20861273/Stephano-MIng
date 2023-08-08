import numpy as np
import torch as T
import random

# last at sample buffer

class ReplayBuffer(object):
    def __init__(self, lidar, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.input_shape = input_shape
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

    def sample_buffer(self, batch_size, sequence_len):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem-sequence_len, batch_size, replace=False)

        # Since the transitions are numpy arrays slicing won't work with the batch list
        # Thus the indices of the batch list are generated below into a list
        # Then instead of slicing a list self.state_memory[starting_index:starting_index+sequence_len]
        # the take function is used

        sequences_indices = [idx + np.arange(sequence_len) for idx in batch]

        states = self.state_memory.take(sequences_indices, axis=0)
        non_image_states = self.non_image_state_memory.take(sequences_indices, axis=0)
        actions = self.action_memory.take(sequences_indices, axis=0)
        rewards = self.reward_memory.take(sequences_indices, axis=0)
        states_ = self.new_state_memory.take(sequences_indices, axis=0)
        new_non_image_state_memory = self.new_non_image_state_memory.take(sequences_indices, axis=0)
        terminal = self.terminal_memory.take(sequences_indices, axis=0)

        return states, non_image_states, actions, rewards, states_, new_non_image_state_memory, terminal