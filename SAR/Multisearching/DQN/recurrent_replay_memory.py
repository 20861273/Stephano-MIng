import numpy as np
import torch as T
import random

# last at sample buffer

class ReplayBuffer(object):
    def __init__(self, guide, max_size, input_shape, nr):
        self.nr = nr
        self.guide = guide
        self.mem_size = max_size
        self.stack_size = 4
        self.non_image_size = 2
        self.mem_cntr = 0 # mem_cntr of the last stored memory
        self.starting_indices = []
        self.image_observations = np.zeros((input_shape),
                                     dtype=np.float32)
        self.image_observations_ = np.zeros((input_shape),
                                     dtype=np.float32)
        self.non_image_observations = np.zeros((self.stack_size, self.non_image_size),
                                     dtype=np.float32)
        self.non_image_observations_ = np.zeros((self.stack_size, self.non_image_size),
                                     dtype=np.float32)
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        if self.guide:
            self.other_state_memory = np.zeros((self.mem_size, self.non_image_size),
                                        dtype=np.float32)
            self.new_other_state_memory = np.zeros((self.mem_size, self.non_image_size),
                                            dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done, other_state=None, other_state_=None, step=None):#image_state, action, reward, image_state_, done, non_image_state, non_image_state_
        index = self.mem_cntr % self.mem_size

        if step == 0: self.starting_indices.append(index)

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.guide:
            self.other_state_memory[index] = other_state
            self.new_other_state_memory[index] = other_state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size, sequence_len):
        max_mem = min(self.mem_cntr, self.mem_size)
        # batch = np.random.choice(max_mem-sequence_len, batch_size, replace=False)

        # don't cross episodes
        # find indices to exclude
        exclude_indices = []
        for s_i in self.starting_indices:
            if s_i == 0: continue
            exclude_indices = exclude_indices + [s_i-j for j in range(sequence_len, 0, -self.nr)]
        # make list of all valid indices
        indices = []
        for i in range(max_mem-sequence_len):
            if i in exclude_indices or i < 0:
                continue
            indices.append(i)
        # choose random indices from valid indices list
        batch = np.random.choice(len(indices), batch_size, replace=False)

        # Since the transitions are numpy arrays slicing won't work with the batch list
        # Thus the indices of the batch list are generated below into a list
        # Then instead of slicing a list self.state_memory[starting_index:starting_index+sequence_len]
        # the take function is used
        # we also make sure to take sequential action from each drone and do not mix the observations 

        sequences_indices = [idx + np.arange(0, sequence_len*self.nr, self.nr) for idx in batch]

        states = self.state_memory.take(sequences_indices, axis=0)
        actions = self.action_memory.take(sequences_indices, axis=0)
        rewards = self.reward_memory.take(sequences_indices, axis=0)
        states_ = self.new_state_memory.take(sequences_indices, axis=0)
        terminal = self.terminal_memory.take(sequences_indices, axis=0)

        if self.guide:
            non_image_states = self.other_state_memory.take(sequences_indices, axis=0)
            new_non_image_state_memory = self.new_other_state_memory.take(sequences_indices, axis=0)

            return states, non_image_states, actions, rewards, states_, new_non_image_state_memory, terminal

        return states, non_image_states, actions, rewards, states_, new_non_image_state_memory, terminal