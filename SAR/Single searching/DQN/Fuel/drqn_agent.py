import numpy as np
import torch as T
from deep_r_q_network import DeepRQNetwork
from replay_memory import ReplayBuffer, PrioritizedReplayMemory
from dqn_environment import Direction, WIDTH, HEIGHT, States
import copy

class DRQNAgent(object):
    def __init__(self, encoding, nr, gamma, epsilon, eps_min, eps_dec, lr, n_actions, starting_beta,
                 input_dims, lidar, c_dims, k_size, s_size, fc_dims,
                 mem_size, batch_size, replace, prioritized=False, algo=None, env_name=None, chkpt_dir='tmp/dqn', device_num=0):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.lidar = lidar
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
        self.encoding = encoding

        global Transition_dtype
        global blank_trans
        if True:
            # Transition_dtype = np.dtype([('timestep', np.int32), ('image_state', np.float32, (self.input_dims)), ('non_image_state', np.float32, (2)), ('action', np.int64), ('reward', np.float32), ('next_image_state', np.float32, (self.input_dims)), ('next_non_image_state', np.float32, (2)), ('done', np.bool_)])
            Transition_dtype = np.dtype([('timestep', np.int32), ('image_state', np.float32, (self.input_dims)), ('non_image_state', np.float32, (1,1)), ('action', np.int64), ('reward', np.float32), ('next_image_state', np.float32, (self.input_dims)), ('next_non_image_state', np.float32, (1,1)), ('done', np.bool_)])

            # blank_trans = (0, np.zeros((self.input_dims), dtype=np.float32), np.zeros((1,2), dtype=np.float32), 0, 0.0,  np.zeros(self.input_dims), np.zeros((1,2), dtype=np.float32), False)
            blank_trans = (0, np.zeros((self.input_dims), dtype=np.float32), np.zeros((1,1), dtype=np.float32), 0, 0.0,  np.zeros(self.input_dims), np.zeros((1,1), dtype=np.float32), False)
        else:
            Transition_dtype = np.dtype([('timestep', np.int32), ('image_state', np.float32, (self.input_dims)), ('action', np.int64), ('reward', np.float32), ('next_image_state', np.float32, (self.input_dims)), ('done', np.bool_)])
            blank_trans = (0, np.zeros((self.input_dims), dtype=np.float32), 0, 0.0,  np.zeros(self.input_dims), False)
        
        # for percentage non-image state
        # Transition_dtype = np.dtype([('timestep', np.int32), ('image_state', np.float32, (self.input_dims)), ('non_image_state', np.float32, (4*nr+1)), ('action', np.int64), ('reward', np.float32), ('next_image_state', np.float32, (self.input_dims)), ('next_non_image_state', np.float32, (4*nr+1)), ('done', np.bool_)])
        # blank_trans = (0, np.zeros((self.input_dims), dtype=np.float32), np.zeros((4*nr+1), dtype=np.float32), 0, 0.0,  np.zeros(self.input_dims), np.zeros((4*nr+1), dtype=np.float32), False)
        
        if self.prioritized:
            # self.memory = ReplayMemory(mem_size, input_dims, n_actions, eps=0.0001, prob_alpha=0.5)
            self.memory = ReplayMemory(self.encoding, self.lidar, max_size=mem_size, batch_size=self.batch_size, replay_alpha=0.5)
            
            # Other implementation of PER in replay_memory.py
            # if self.lidar and self.encoding == "image":
            #     lidar = True
            # else:
            #     lidar = False
                
            # self.memory = PrioritizedReplayMemory(lidar, self.input_dims, mem_size, alpha=0.5)

        else:
            self.memory = ReplayBuffer(self.lidar, mem_size, input_dims, n_actions)
        
        self.q_eval = DeepRQNetwork(self.encoding, nr, self.lr, self.n_actions,
                                    self.input_dims, self.lidar,
                                    c_dims, k_size, s_size,
                                    fc_dims,
                                    device_num,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepRQNetwork(self.encoding, nr, self.lr, self.n_actions,
                                    self.input_dims, self.lidar,
                                    c_dims, k_size, s_size, fc_dims,
                                    device_num,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, env, i_r, image_observation, non_image_observation, allow_windowed_revisiting, previous_action=None):
        if np.random.random() > self.epsilon or not self.q_eval.training:
            image_state = T.tensor(np.array([image_observation]),dtype=T.float32).to(self.q_eval.device)
            if True:
                non_image_state = T.tensor(np.array([non_image_observation]),dtype=T.float32).to(self.q_eval.device)
                self.q_eval.init_hidden(1)
                actions = self.q_eval.forward(image_state, non_image_state)
            else:
                actions = self.q_eval.forward(image_state)
            actions = self.check_obstacles(env, i_r, actions.tolist())
            action = T.argmax(actions).item()
        else:
            actions = self.check_obstacles(env, i_r, [self.action_space])
            exclude = [i for i,e in enumerate(actions.tolist()[0]) if e == float("-inf")]
            action = np.random.choice([i for i in range(len(self.action_space)) if i not in exclude])

        
        return action

    def check_obstacles(self, env, i_r, actions):
        surroundings = []
        right_is_boundary = env.pos[i_r].x == WIDTH - 1
        left_is_boundary = env.pos[i_r].x == 0
        top_is_boundary = env.pos[i_r].y == 0
        bottom_is_boundary = env.pos[i_r].y == HEIGHT - 1

        surroundings.append(right_is_boundary or (env.grid[env.pos[i_r].y][env.pos[i_r].x+1] == States.OBS.value if not right_is_boundary else True))
        surroundings.append(left_is_boundary or (env.grid[env.pos[i_r].y][env.pos[i_r].x-1] == States.OBS.value if not left_is_boundary else True))
        surroundings.append(top_is_boundary or (env.grid[env.pos[i_r].y-1][env.pos[i_r].x] == States.OBS.value if not top_is_boundary else True))
        surroundings.append(bottom_is_boundary or (env.grid[env.pos[i_r].y+1][env.pos[i_r].x] == States.OBS.value if not bottom_is_boundary else True))

        temp_actions = []
        for i in range(len(actions[0])):
            if not surroundings[i]: temp_actions.append(actions[0][i])
            else: temp_actions.append(float("-inf"))
        
        return T.tensor([temp_actions])
    
    def store_transition(self, image_state, non_image_state, action, reward, image_state_, non_image_state_, done):
        if True:
            self.memory.store_transition(image_state, action, reward, image_state_, done, non_image_state, non_image_state_)
            # self.memory.add(image_state, action, reward, image_state_, done, non_image_state, non_image_state_)
        else:
            self.memory.store_transition(image_state, action, reward, image_state_, done)
            # self.memory.add(image_state, action, reward, image_state_, done)

    def sample_memory(self, beta, prioritized=False):
        if not prioritized:
            image_state, non_image_state, action, reward, new_image_state, new_non_image_state, done = \
                                    self.memory.sample_buffer(self.batch_size)
            
            image_states = T.tensor(image_state).to(self.q_eval.device)
            
            rewards = T.tensor(reward).to(self.q_eval.device)
            dones = T.tensor(done).to(self.q_eval.device)
            actions = T.tensor(action).to(self.q_eval.device)
            image_states_ = T.tensor(new_image_state).to(self.q_eval.device)
            

            if True:
                non_image_states = T.tensor(non_image_state).to(self.q_eval.device)
                non_image_states_ = T.tensor(new_non_image_state).to(self.q_eval.device)

                return image_states, non_image_states, actions, rewards, image_states_, non_image_states_, dones
            else:
                return image_states, actions, rewards, image_states_, dones
            
        elif prioritized:
            image_states, non_image_state, actions, rewards, image_states_, non_image_states_, dones, weights = \
                                    self.memory.sample_buffer(self.gamma, self.batch_size, self.q_eval, self.q_next, beta)
            
            weights = T.tensor(weights).to(self.q_eval.device)
            
            return image_states, non_image_state, actions, rewards, image_states_, non_image_states_, dones, weights
    

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
            indices = np.arange(self.batch_size)
            
            tree_idxs, data, weights = self.memory.sample(self.beta)
            image_states=T.tensor(np.copy(data[:]['image_state'])).to(self.q_eval.device)
            rewards=T.tensor(np.copy(data[:]['reward'])).to(self.q_eval.device)
            dones=T.tensor(np.copy(data[:]['done'])).to(self.q_eval.device)
            actions=T.tensor(np.copy(data[:]['action'])).to(self.q_eval.device)
            image_states_=T.tensor(np.copy(data[:]['next_image_state'])).to(self.q_eval.device)

            if True:
                non_image_states=T.tensor(np.copy(data[:]['non_image_state'])).to(self.q_eval.device)
                non_image_states = non_image_states.reshape((-1, 1))
                non_image_states_=T.tensor(np.copy(data[:]['next_non_image_state'])).to(self.q_eval.device)
                non_image_states_ = non_image_states_.reshape((-1, 1))
                
                self.q_eval.init_hidden(self.batch_size)
                q_pred = self.q_eval.forward(image_states, non_image_states)[indices, actions]
                self.q_next.init_hidden(self.batch_size)
                q_next = self.q_next.forward(image_states_, non_image_states_).max(dim=1)[0]
            else:
                q_pred = self.q_eval.forward(image_states)[indices, actions]
                q_next = self.q_next.forward(image_states_).max(dim=1)[0]           

            q_next[dones] = 0.0
            q_target = rewards + self.gamma*q_next
            errors = T.sub(q_target, q_pred).to(self.q_eval.device)
            loss = self.q_eval.loss(T.multiply(errors, T.tensor(weights).to(self.q_eval.device)).float(), T.zeros(self.batch_size).to(self.q_eval.device).float()).to(self.q_eval.device)

        else:
            # add zero grad before adding trasition
            self.q_eval.optimizer.zero_grad()
            # replace target here
            self.replace_target_network()

            
            indices = np.arange(self.batch_size)
            if True:
                image_states, non_image_states, actions, rewards, image_states_, non_image_states_, dones = self.sample_memory(self.beta)
                self.q_eval.init_hidden(self.batch_size)
                q_pred = self.q_eval.forward(image_states, non_image_states)[indices, actions]
                self.q_next.init_hidden(self.batch_size)
                q_next = self.q_next.forward(image_states_, non_image_states_).max(dim=1)[0]
            else:
                image_states, actions, rewards, image_states_, dones = self.sample_memory(self.beta)
                q_pred = self.q_eval.forward(image_states)[indices, actions]
                q_next = self.q_next.forward(image_states_).max(dim=1)[0]

            q_next[dones] = 0.0
            q_target = rewards + self.gamma*q_next
            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)


        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1

        self.decrement_epsilon()

        if self.prioritized: self.memory.update_priorities(tree_idxs, errors)

        return loss.cpu().detach().numpy()


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
    def __init__(self, encoding, lidar, max_size, batch_size, replay_alpha):
        self.encoding = encoding
        self.lidar = lidar
        self.batch_size=batch_size
        self.capacity = max_size
        self.replay_alpha = replay_alpha
        self.transitions = SegmentTree(self.capacity)
        self.t = 0

    def store_transition(self, image_state, action, reward, next_image_state, done, non_image_state=None, next_non_image_state=None):
        if True:
            self.transitions.append((self.t, image_state, non_image_state, action, reward, next_image_state, next_non_image_state, done), self.transitions.max)  # Store new transition with maximum priority
        else:
            self.transitions.append((self.t, image_state, action, reward, next_image_state, done), self.transitions.max)  # Store new transition with maximum priority
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
            print('Capacity is 0')

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

    def update_priorities(self, idxs, priorities, epsilon=0.001):
        priorities = np.power(np.abs(priorities.cpu().detach().numpy())+epsilon, self.replay_alpha)
        self.transitions.update(idxs, priorities)
        if np.any(priorities==0):
            print('Priorities are 0')
