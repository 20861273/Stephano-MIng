import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from collections import deque
import random
import time

#Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.float32, (state_shape)), ('action', np.int64), ('reward', np.float32), ('next_state', np.float32, (state_shape)), ('done', np.bool_)])
blank_trans = (0, np.zeros((12), dtype=np.float32), 0, 0.0,  np.zeros(12), False)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

# no tree
class PrioritisedReplayBuffer(object):
    def __init__(self, max_size, input_shape, batch_size, replay_alpha):
        self.mem_size = max_size
        self.replay_alpha = replay_alpha
        self.mem_cntr = 0
        self.curr_mem_pos = 0
        self.batch_size = batch_size
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.priorities = np.zeros(self.mem_size, dtype=np.float32)
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        #self.priorities.append(max(self.priorities, default=1))
        
        if self.curr_mem_pos<self.batch_size:
            self.priorities[index] = 1
        else:
            self.priorities[index] = max(self.priorities, default=1)
        
        self.mem_cntr += 1
        self.curr_mem_pos = min(self.mem_cntr, self.mem_size)
    
    def get_probabilities(self):
        priorities = self.priorities[0:self.curr_mem_pos]
        scaled_priorities = np.power(np.array(priorities), self.replay_alpha)
        sample_probabilities = np.divide(scaled_priorities, np.sum(scaled_priorities))
        return sample_probabilities
    
    def get_importance(self, probabilities, replay_beta):
        importance = np.power(np.multiply(np.divide(1, self.curr_mem_pos), np.divide(1, probabilities)), replay_beta)
        importance_normalized = np.divide(importance, np.max(importance))
        return importance_normalized

    def sample_buffer(self, batch_size, replay_beta):
        sample_size = min(self.mem_cntr, batch_size)
        sample_probs = self.get_probabilities()
        #sample_indices = random.choices(range(len(self.priorities)), k=sample_size, weights=sample_probs)
        sample_indices = random.choices(range(self.curr_mem_pos), k=sample_size, weights=sample_probs)
        
        states = self.state_memory[sample_indices]
        actions = self.action_memory[sample_indices]
        rewards = self.reward_memory[sample_indices]
        states_ = self.new_state_memory[sample_indices]
        terminal = self.terminal_memory[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices], replay_beta)

        return states, actions, rewards, states_, terminal, importance, sample_indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        for i,e in zip(indices, errors):
            self.priorities[i] = abs(e.item()) + offset

class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([blank_trans] * size, dtype=Transition_dtype)
        self.max = 1  # Initial max value to return (1 = 1^ω)

    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)
    
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)
    
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)
    
    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)
    
    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]
    

class ReplayMemory:
    def __init__(self, max_size, batch_size, replay_alpha):
        self.batch_size=batch_size
        self.capacity = max_size
        self.replay_alpha = replay_alpha
        self.transitions = SegmentTree(self.capacity)
        self.t = 0

    def store_transition(self, state, action, reward, next_state, done):
        self.transitions.append((self.t, state, action, reward, next_state, done), self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if done else self.t + 1  # Start new episodes with t = 0
    
    def sample(self, replay_beta):
        capacity = self.capacity if self.transitions.full else self.transitions.index
        while True:
            p_total=self.transitions.total()
            samples = np.random.uniform(0, p_total, self.batch_size)
            probs, data_idxs, tree_idxs = self.transitions.find(samples)
            if np.all(data_idxs<=capacity):
                break
        
        data = self.transitions.get(data_idxs)
        probs = probs / p_total
        #weights = (capacity * probs) ** -replay_beta  # Compute importance-sampling weights w
        #weights = weights / weights.max()  # Normalise by max importance-sampling weight from batch
        
        if np.any(probs==0):
            print('Probs are 0')
        if capacity==0:
            print('Probs are 0')

        weights = np.power(np.multiply(np.divide(1, capacity), np.divide(1, probs)), replay_beta)
        if np.any(weights==np.inf):
            print('weights are inf')
        if np.any(weights==0):
            print('weights are 0')
        
        norm_weights = np.divide(weights, np.max(weights))
        if np.max(weights)==np.inf:
            print('weights are inf')
        if np.max(weights)==0:
            print('weights are 0')

        #return tree_idxs, states, actions, returns, next_states, nonterminals, weights
        return tree_idxs, data, norm_weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(np.abs(priorities.cpu().detach().numpy()), self.replay_alpha)
        self.transitions.update(idxs, priorities)


class DuelingLinearDeepQNetwork(nn.Module):
    def __init__(self, ALPHA, n_actions, input_dims, fc1_dims, fc2_dims):
        super(DuelingLinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        #self.checkpoint_dir = chkpt_dir
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_dqn')

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        V = self.V(l2)
        A = self.A(l2)

        return V, A

class DuelingLinearCategoricalDeepQNetwork(nn.Module):
    def __init__(self, ALPHA, n_actions, input_dims, fc1_dims, fc2_dims, n_atoms):
        super(DuelingLinearDeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1*n_atoms)
        self.A = nn.Linear(fc2_dims, n_actions*n_atoms)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        #self.checkpoint_dir = chkpt_dir
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_dqn')

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        V = self.V(l2)
        A = self.A(l2)

        return V, A



class agent(object):

    def __init__(self, agent_dict):
        
        self.agent_dict = agent_dict
        self.name = self.agent_dict['name']
        self.gamma = self.agent_dict['gamma']
        self.epsilon = self.agent_dict['epsilon']
        self.eps_end = self.agent_dict['eps_end']
        self.eps_dec = self.agent_dict['eps_dec']
        self.action_space = [i for i in range(self.agent_dict['n_actions'])]
        self.learn_step_counter = 0
        self.batch_size = self.agent_dict['batch_size']
        self.replace_target_cnt = self.agent_dict['replace']

        self.n_atoms = 50
        self.v_min = -10
        self.v_max = 10
        self.delta = (self.v_max - self.v_min)/self.n_atoms
        self.df = self.gamma

        self.atoms = T.arange(self.v_min, self.v_max, self.delta).unsqueeze(0)


        global Transition_dtype
        global blank_trans 
        Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.float32, (self.agent_dict['input_dims'])), ('action', np.int64), ('reward', np.float32), ('next_state', np.float32, (self.agent_dict['input_dims'])), ('done', np.bool_)])
        blank_trans = (0, np.zeros((12), dtype=np.float32), 0, 0.0,  np.zeros(12), False)
        #self.memory = PrioritisedReplayBuffer(self.agent_dict['max_mem_size'], self.agent_dict['input_dims'], self.agent_dict['batch_size'], self.agent_dict['replay_alpha'])
        
        self.memory = ReplayMemory(max_size=self.agent_dict['max_mem_size'], batch_size=self.agent_dict['batch_size'], replay_alpha=self.agent_dict['replay_alpha'])

        self.q_eval = DuelingLinearDeepQNetwork(self.agent_dict['alpha'], self.agent_dict['n_actions'], input_dims=self.agent_dict['input_dims'],
                                                fc1_dims=self.agent_dict['fc1_dims'], fc2_dims=self.agent_dict['fc2_dims'])

        self.q_next = DuelingLinearDeepQNetwork(self.agent_dict['alpha'], self.agent_dict['n_actions'], input_dims=self.agent_dict['input_dims'],
                                                fc1_dims=self.agent_dict['fc1_dims'], fc2_dims=self.agent_dict['fc2_dims'])
     

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            
            #start = time.time()
            state = T.tensor(observation, dtype=T.float32).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
            #end = time.time()
            #print(end-start)
        
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrease_epsilon(self):
        if self.epsilon > self.eps_end:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_end

    def learn(self, replay_beta):
        if self.memory.transitions.index<self.batch_size and self.memory.transitions.full==False:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        tree_idxs, data, weights = self.memory.sample(replay_beta)

        states=T.tensor(np.copy(data[:]['state'])).to(self.q_eval.device)
        rewards=T.tensor(np.copy(data[:]['reward'])).to(self.q_eval.device)
        dones=T.tensor(np.copy(data[:]['done'])).to(self.q_eval.device)
        actions=T.tensor(np.copy(data[:]['action'])).to(self.q_eval.device)
        states_=T.tensor(np.copy(data[:]['next_state'])).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))
        
        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]
        errors = T.sub(q_target, q_pred).to(self.q_eval.device)
        loss = self.q_eval.loss(T.multiply(errors, T.tensor(weights).to(self.q_eval.device)).float(), T.zeros(64).to(self.q_eval.device).float()).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.memory.update_priorities(tree_idxs, errors)
    
    def learn_distributional_rl(self, replay_beta):
        if self.memory.transitions.index<=(self.batch_size+1) and self.memory.transitions.full==False:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        tree_idxs, data, weights = self.memory.sample(replay_beta)

        states=T.tensor(np.copy(data[:]['state'])).to(self.q_eval.device)
        rewards=T.tensor(np.copy(data[:]['reward'])).to(self.q_eval.device)
        dones=T.tensor(np.copy(data[:]['done'])).to(self.q_eval.device)
        actions=T.tensor(np.copy(data[:]['action'])).to(self.q_eval.device)
        states_=T.tensor(np.copy(data[:]['next_state'])).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = F.log_softmax(T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions])
        q_next = F.log_softmax(T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))))
        q_eval = F.log_softmax(T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True))))
        
        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]
        errors = T.sub(q_target, q_pred).to(self.q_eval.device)
        loss = self.q_eval.loss(T.multiply(errors, T.tensor(weights).to(self.q_eval.device)).float(), T.zeros(64).to(self.q_eval.device).float()).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.memory.update_priorities(tree_idxs, errors)

    def compute_targets(self, rewards, next_states, dones):
        mask = np.invert(dones.astype(bool))
        atoms = T.arange(self.v_min, self.v_max, self.delta)
        atoms = (rewards + self.df * dones[:, None] * atoms).clamp(min=self.v_min, max=self.v_max)
        b = (atoms - self.v_min) / self.delta
        l = T.floor(b).long()
        u = T.ceil(b).clamp(max=self.n_atoms - 1).long()
        with T.no_grad():
            v, a = self.q_next(next_states)
            z_prime = F.log_softmax(T.add(v, (a - a.mean(dim=1, keepdim=True))))
        target_actions = T.argmax(a, dim=1)
        
        z_prime = T.cat([z_prime[i, target_actions[i]] for i in range(z_prime.shape[0])])
        z_prime = T.cat([z_prime[i, target_actions[i]] for i in range(z_prime.shape[0])])

        # For elements that do not have a next state, atoms are all equal to reward and we set a
        # uniform distribution (it will collapse to the same atom in any case)
        probabilities = T.ones((self.batch_size, self.n_atoms)) / self.n_atoms
        probabilities[mask] = z_prime
        # Compute partitions of atoms
        lower = probabilities * (u - b)
        upper = probabilities * (b - l)
        z_projected = T.zeros_like(probabilities)
        z_projected.scatter_add_(1, l, lower)
        z_projected.scatter_add_(1, u, upper)
        return z_projected

    def save_agent(self):
        T.save(self.q_eval.state_dict(), 'agents/' + self.name + 'q_eval_weights')
        T.save(self.q_next.state_dict(), 'agents/' + self.name + 'q_next_weights')
        self.agent_dict['epsilon'] = self.epsilon
        
        outfile = open('agents/' + self.name + '_hyper_parameters', 'wb')
        pickle.dump(self.agent_dict, outfile)
        outfile.close()

    def load_weights(self, name):
        self.q_eval.load_state_dict(T.load('agents/' + name + 'q_eval_weights'))
        self.q_eval.load_state_dict(T.load('agents/' + name + 'q_next_weights'))
