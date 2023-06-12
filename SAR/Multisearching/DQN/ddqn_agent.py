import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer, PrioritizedReplayMemory

class DDQNAgent(object):
    def __init__(self, nr, gamma, epsilon, eps_min, eps_dec, lr, n_actions, starting_beta,
                 input_dims, c_dims, k_size, s_size, fc_dims,
                 mem_size, batch_size, replace, prioritized=False, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.prioritized = prioritized
        self.beta = starting_beta

        global Transition_dtype
        global blank_trans 
        Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.float32, (self.input_dims)), ('action', np.int64), ('reward', np.float32), ('next_state', np.float32, (self.input_dims)), ('done', np.bool_)])
        blank_trans = (0, np.zeros((self.input_dims), dtype=np.float32), 0, 0.0,  np.zeros(self.input_dims), False)
        

        if self.prioritized:
            # self.memory = ReplayMemory(mem_size, input_dims, n_actions, eps=0.0001, prob_alpha=0.5)
            self.memory = ReplayMemory(max_size=mem_size, batch_size=self.batch_size, replay_alpha=0.5)

        else:
            self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        self.q_eval = DeepQNetwork(nr, self.lr, self.n_actions,
                                    self.input_dims,
                                    c_dims, k_size, s_size,
                                    fc_dims,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(nr, self.lr, self.n_actions,
                                    self.input_dims,
                                    c_dims, k_size, s_size, fc_dims,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon or not self.q_eval.training:
            state = T.tensor(np.array([observation]),dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self, beta=0, prioritized=False):
        if not prioritized:
            state, action, reward, new_state, done = \
                                    self.memory.sample_buffer(self.batch_size)
            
            states = T.tensor(state).to(self.q_eval.device)
            rewards = T.tensor(reward).to(self.q_eval.device)
            dones = T.tensor(done).to(self.q_eval.device)
            actions = T.tensor(action).to(self.q_eval.device)
            states_ = T.tensor(new_state).to(self.q_eval.device)

            return states, actions, rewards, states_, dones
        elif prioritized:
            states, actions, rewards, states_, dones, weights = \
                                    self.memory.sample_buffer(self.gamma, self.batch_size, self.q_eval, self.q_next, beta)
            
            # states = T.tensor(states).to(self.q_eval.device)
            # rewards = T.tensor(rewards).to(self.q_eval.device)
            # dones = T.tensor(dones).to(self.q_eval.device)
            # actions = T.tensor(actions).to(self.q_eval.device)
            # states_ = T.tensor(states_).to(self.q_eval.device)
            weights = T.tensor(weights).to(self.q_eval.device)
            
            return states, actions, rewards, states_, dones, weights
    

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self, i_exp, i_ts, i_episode, time, loss):
        self.q_eval.save_checkpoint(i_exp, i_ts, i_episode, time, loss)
        self.q_next.save_checkpoint(i_exp, i_ts, i_episode, time, loss)

    def load_models(self):
        checkpoint = self.q_eval.load_checkpoint()
        checkpoint = self.q_next.load_checkpoint()

        return checkpoint

    def learn(self):
        if self.prioritized:
            if self.memory.transitions.index<self.batch_size and self.memory.transitions.full==False:
                return
        else:
            if self.memory.mem_cntr < self.batch_size:
                return

        if self.prioritized:
            # add zero grad before adding trasition
            self.q_eval.optimizer.zero_grad()
            # replace target here
            self.replace_target_network()

            self.beta += 0.00005
            self.beta = min(1, self.beta)
            # states, actions, rewards, states_, dones, weights = self.sample_memory(self.beta, self.prioritized)
            tree_idxs, data, weights = self.memory.sample(self.beta)
            states=T.tensor(np.copy(data[:]['state'])).to(self.q_eval.device)
            rewards=T.tensor(np.copy(data[:]['reward'])).to(self.q_eval.device)
            dones=T.tensor(np.copy(data[:]['done'])).to(self.q_eval.device)
            actions=T.tensor(np.copy(data[:]['action'])).to(self.q_eval.device)
            states_=T.tensor(np.copy(data[:]['next_state'])).to(self.q_eval.device)

            indices = np.arange(self.batch_size)
            q_pred = self.q_eval.forward(states)[indices, actions]
            q_next = self.q_next.forward(states_).max(dim=1)[0]

            q_next[dones] = 0.0
            q_target = rewards + self.gamma*q_next
            errors = T.sub(q_target, q_pred).to(self.q_eval.device)
            loss = self.q_eval.loss(T.multiply(errors, T.tensor(weights).to(self.q_eval.device)).float(), T.zeros(self.batch_size).to(self.q_eval.device).float()).to(self.q_eval.device)
        

            # td_error = self.memory.get_td_error(self.batch_size, self.gamma, self.q_eval, self.q_next, states, states_, actions, rewards, dones)
            # loss = pow(td_error, 2) * weights
            # loss = loss.mean()
        else:
            # add zero grad before adding trasition
            self.q_eval.optimizer.zero_grad()
            # replace target here
            self.replace_target_network()

            states, actions, rewards, states_, dones = self.sample_memory()

            indices = np.arange(self.batch_size)

            q_pred = self.q_eval.forward(states)[indices, actions]
            q_next = self.q_next.forward(states_)
            q_eval = self.q_eval.forward(states_)

            max_actions = T.argmax(q_eval, dim=1)

            q_next[dones] = 0.0

            q_target = rewards + self.gamma*q_next[indices, max_actions]
            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        # self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        # self.replace_target_network()

        self.learn_step_counter += 1

        self.decrement_epsilon()

        if self.prioritized: self.memory.update_priorities(tree_idxs, errors)

        return loss


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
