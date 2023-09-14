import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer, PrioritizedReplayMemory
from dqn_new_environment import Direction, WIDTH, HEIGHT, States, Point
import copy

class DQNAgent(object):
    def __init__(self, encoding, nr, gamma, epsilon, eps_min, eps_dec, lr, n_actions, starting_beta,
                 input_dims, guide, lidar, lstm, c_dims, k_size, s_size, fc_dims,
                 mem_size, batch_size, replace, nstep, nstep_N, prioritized=False, algo=None, env_name=None, chkpt_dir='tmp/dqn', device_num=0):
        self.nr = nr
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.guide = guide
        self.lidar = lidar
        self.lstm = lstm
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
        self.nstep = nstep
        self.nstep_N = nstep_N

        global Transition_dtype
        global blank_trans
        if (self.lidar or self.guide) and "image" in self.encoding:
            Transition_dtype = np.dtype([('timestep', np.int32), ('image_state', np.float32, (self.input_dims)), ('non_image_state', np.float32, (2)), ('action', np.int64), ('reward', np.float32), ('next_image_state', np.float32, (self.input_dims)), ('next_non_image_state', np.float32, (2)), ('done', np.bool_)])
            blank_trans = (0, np.zeros((self.input_dims), dtype=np.float32), np.zeros((2), dtype=np.float32), 0, 0.0,  np.zeros(self.input_dims), np.zeros((2), dtype=np.float32), False)
        else:
            Transition_dtype = np.dtype([('timestep', np.int32), ('image_state', np.float32, (self.input_dims)), ('action', np.int64), ('reward', np.float32), ('next_image_state', np.float32, (self.input_dims)), ('done', np.bool_)])
            blank_trans = (0, np.zeros((self.input_dims), dtype=np.float32), 0, 0.0,  np.zeros(self.input_dims), False)
        
        # for percentage non-image state
        # Transition_dtype = np.dtype([('timestep', np.int32), ('image_state', np.float32, (self.input_dims)), ('non_image_state', np.float32, (4*nr+1)), ('action', np.int64), ('reward', np.float32), ('next_image_state', np.float32, (self.input_dims)), ('next_non_image_state', np.float32, (4*nr+1)), ('done', np.bool_)])
        # blank_trans = (0, np.zeros((self.input_dims), dtype=np.float32), np.zeros((4*nr+1), dtype=np.float32), 0, 0.0,  np.zeros(self.input_dims), np.zeros((4*nr+1), dtype=np.float32), False)
        
        if self.prioritized:
            # self.memory = ReplayMemory(mem_size, input_dims, n_actions, eps=0.0001, prob_alpha=0.5)
            self.memory = ReplayMemory(self.nr, self.encoding, self.lidar, self.guide, mem_size, self.batch_size, self.nstep, self.nstep_N, replay_alpha=0.5)
            
            # Other implementation of PER in replay_memory.py
            # if self.lidar and self.encoding == "image":
            #     lidar = True
            # else:
            #     lidar = False
                
            # self.memory = PrioritizedReplayMemory(lidar, self.input_dims, mem_size, alpha=0.5)

        else:
            self.memory = ReplayBuffer(self.nr, self.guide, mem_size, input_dims, self.nstep, self.nstep_N)

        self.q_eval = DeepQNetwork(self.encoding, nr, self.lr, self.n_actions,
                                    self.input_dims, self.guide, self.lidar, self.lstm,
                                    c_dims, k_size, s_size,
                                    fc_dims,
                                    device_num,
                                    name=self.env_name+'_'+self.algo+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.encoding, nr, self.lr, self.n_actions,
                                    self.input_dims, self.guide, self.lidar, self.lstm,
                                    c_dims, k_size, s_size, fc_dims,
                                    device_num,
                                    name=self.env_name+'_'+self.algo+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def choose_action(self, env, i_r, image_observation, non_image_observation, allow_windowed_revisiting, stacked, other_actions=None, previous_action=None):
        chose_max_action = False
        if np.random.random() > self.epsilon or not self.q_eval.training:
            chose_max_action = True
            if stacked: image_state = T.tensor(np.array(image_observation),dtype=T.float32).to(self.q_eval.device)
            else: image_state = T.tensor(np.array([image_observation]),dtype=T.float32).to(self.q_eval.device)
            if (self.lidar or self.guide) and "image" in self.encoding:
                # non_image_state = T.tensor(np.array([non_image_observation]),dtype=T.float32).to(self.q_eval.device).reshape((1,1))
                if stacked: non_image_state = T.tensor(np.array(non_image_observation),dtype=T.float32).to(self.q_eval.device)
                else: non_image_state = T.tensor(np.array([non_image_observation]),dtype=T.float32).to(self.q_eval.device)
                actions = self.q_eval.forward(image_state, non_image_state)
            else:
                actions = self.q_eval.forward(image_state)
            valid_actions = self.check_obstacles(env, i_r, other_actions, actions.tolist())
            if T.all(valid_actions == -float('inf')):
                return T.argmax(actions).item(), True
            action = T.argmax(valid_actions).item()
        else:
            valid_actions = self.check_obstacles(env, i_r, other_actions, [self.action_space])
            if T.all(valid_actions == -float('inf')):
                return np.random.choice(self.action_space), True
            exclude = [i for i,e in enumerate(valid_actions.tolist()[0]) if e == float("-inf")]
            action = np.random.choice([i for i in range(len(self.action_space)) if i not in exclude])
            # action = np.random.choice(self.action_space)
        
        # if not previous_action == None and not allow_windowed_revisiting:
        #     if action == Direction.UP.value and previous_action == Direction.DOWN.value:
        #         if not chose_max_action:
        #             image_state = T.tensor(np.array([image_observation]),dtype=T.float32).to(self.q_eval.device)
        #             if self.lidar and "image" in self.encoding:
        #                 non_image_state = T.tensor(np.array([non_image_observation]),dtype=T.float32).to(self.q_eval.device)
        #                 actions = self.q_eval.forward(image_state, non_image_state)
        #             else:
        #                 actions = self.q_eval.forward(image_state)
        #         actions = T.topk(actions, 2)
        #         action = actions.indices[0][1].item()
        #     if action == Direction.DOWN.value and previous_action == Direction.UP.value:
        #         if not chose_max_action:
        #             image_state = T.tensor(np.array([image_observation]),dtype=T.float32).to(self.q_eval.device)
        #             if self.lidar and "image" in self.encoding:
        #                 non_image_state = T.tensor(np.array([non_image_observation]),dtype=T.float32).to(self.q_eval.device)
        #                 actions = self.q_eval.forward(image_state, non_image_state)
        #             else:
        #                 actions = self.q_eval.forward(image_state)
        #         actions = T.topk(actions, 2)
        #         action = actions.indices[0][1].item()
        #     if action == Direction.RIGHT.value and previous_action == Direction.LEFT.value:
        #         if not chose_max_action:
        #             image_state = T.tensor(np.array([image_observation]),dtype=T.float32).to(self.q_eval.device)
        #             if self.lidar and "image" in self.encoding:
        #                 non_image_state = T.tensor(np.array([non_image_observation]),dtype=T.float32).to(self.q_eval.device)
        #                 actions = self.q_eval.forward(image_state, non_image_state)
        #             else:
        #                 actions = self.q_eval.forward(image_state)
        #         actions = T.topk(actions, 2)
        #         action = actions.indices[0][1].item()
        #     if action == Direction.LEFT.value and previous_action == Direction.RIGHT.value:
        #         if not chose_max_action:
        #             image_state = T.tensor(np.array([image_observation]),dtype=T.float32).to(self.q_eval.device)
        #             if self.lidar and "image" in self.encoding:
        #                 non_image_state = T.tensor(np.array([non_image_observation]),dtype=T.float32).to(self.q_eval.device)
        #                 actions = self.q_eval.forward(image_state, non_image_state)
        #             else:
        #                 actions = self.q_eval.forward(image_state)
        #         actions = T.topk(actions, 2)
        #         action = actions.indices[0][1].item()
        
        return action, False

    def check_obstacles(self, env, i_r, other_actions, actions):
        surroundings = [False]*(len(self.action_space))

        right_is_boundary = env.pos[i_r].x == WIDTH - 1
        left_is_boundary = env.pos[i_r].x == 0
        top_is_boundary = env.pos[i_r].y == 0
        bottom_is_boundary = env.pos[i_r].y == HEIGHT - 1

        # all possible positions of current drone
        # also check boundaries
        possible_positions = [False]*len(self.action_space)
        current_temp_x = env.pos[i_r].x
        current_temp_y = env.pos[i_r].y
        if env.direction[i_r] == "up": #up
            # forward action
            if not top_is_boundary:
                possible_positions[0] = Point(current_temp_x, current_temp_y-1)
            else:
                surroundings[0] = True
            # left action
            if not left_is_boundary:
                possible_positions[2] = Point(current_temp_x-1, current_temp_y)
            else:
                surroundings[2] = True
            # right action
            if not right_is_boundary:
                possible_positions[1] = Point(current_temp_x+1, current_temp_y)
            else:
                surroundings[1] = True
        if env.direction[i_r] == "down": #down
            # forward action
            if not bottom_is_boundary:
                possible_positions[0] = Point(current_temp_x, current_temp_y+1)
            else:
                surroundings[0] = True
            # right action
            if not left_is_boundary:
                possible_positions[1] = Point(current_temp_x-1, current_temp_y)
            else:
                surroundings[1] = True
            # left action
            if not right_is_boundary:
                possible_positions[2] = Point(current_temp_x+1, current_temp_y)
            else:
                surroundings[2] = True
        if env.direction[i_r] == "right": #right
            # left action
            if not top_is_boundary:
                possible_positions[2] = Point(current_temp_x, current_temp_y-1)
            else:
                surroundings[2] = True
            # forward action
            if not right_is_boundary:
                possible_positions[0] = Point(current_temp_x+1, current_temp_y)
            else:
                surroundings[0] = True
            # right action
            if not bottom_is_boundary:
                possible_positions[1] = Point(current_temp_x, current_temp_y+1)
            else:
                surroundings[1] = True
        if env.direction[i_r] == "left": #left
            # right action
            if not top_is_boundary:
                possible_positions[1] = Point(current_temp_x, current_temp_y-1)
            else:
                surroundings[1] = True
            # forward action
            if not left_is_boundary:
                possible_positions[0] = Point(current_temp_x-1, current_temp_y)
            else:
                surroundings[0] = True
            # left action
            if not bottom_is_boundary:
                possible_positions[2] = Point(current_temp_x, current_temp_y+1)
            else:
                surroundings[2] = True

        # check collision with obstacles
        for i in range(len(possible_positions)-1):
            if not possible_positions[i]: continue
            if env.grid[possible_positions[i].y, possible_positions[i].x] == States.OBS.value: surroundings[i] = True

        # move drones to new positions
        new_positions = [False]*self.nr
        for r in range(self.nr):
            temp_x = env.pos[r].x
            temp_y = env.pos[r].y
            if r >= i_r: break
            if other_actions[r] == None: break
            if env.direction[r] == "up":
                if other_actions[r] == Direction.FORWARD.value:
                    temp_y -= 1
                if other_actions[r] == Direction.RIGHT.value:
                    temp_x += 1
                if other_actions[r] == Direction.LEFT.value:
                    temp_x -= 1
            elif env.direction[r] == "down":
                if other_actions[r] == Direction.FORWARD.value:
                    temp_y += 1
                if other_actions[r] == Direction.RIGHT.value:
                    temp_x -= 1
                if other_actions[r] == Direction.LEFT.value:
                    temp_x += 1
            elif env.direction[r] == "right":
                if other_actions[r] == Direction.FORWARD.value:
                    temp_x += 1
                if other_actions[r] == Direction.RIGHT.value:
                    temp_y += 1
                if other_actions[r] == Direction.LEFT.value:
                    temp_y -= 1
            elif env.direction[r] == "left":
                if other_actions[r] == Direction.FORWARD.value:
                    temp_x -= 1
                if other_actions[r] == Direction.RIGHT.value:
                    temp_y -= 1
                if other_actions[r] == Direction.LEFT.value:
                    temp_y += 1

            new_positions[r] = Point(temp_x, temp_y)

        # check if drones collide
        for r in range(self.nr):
            if r >= i_r: break
            if other_actions[r] == None: break
            
            for i in range(len(possible_positions)):
                # checks new positions
                if new_positions[r] == possible_positions[i]: surroundings[i] = True
                # checks crossing positions
                if possible_positions[i] == env.pos[r] and new_positions[r] == env.pos[i_r]:
                    surroundings[i] = True

        temp_actions = []
        for i in range(len(actions[0])):
            if not surroundings[i]: temp_actions.append(actions[0][i])
            else: temp_actions.append(float("-inf"))
        
        return T.tensor([temp_actions])
    
    def store_transition(self, image_state, non_image_state, action, reward, image_state_, non_image_state_, done):
        # rotates observations for 3 extra states in memory
        # tuples = []
        # tuples.append((image_state, non_image_state, action, reward, image_state_, non_image_state_, done))
        # temp_image_state = np.copy(image_state)
        # temp_image_state_ = np.copy(image_state_)
        # for _ in range(3):
        #     breakpoint
        #     for i in range(image_state.shape[0]):
        #         temp_image_state[i] = np.rot90(temp_image_state[i])
        #         temp_image_state_[i] = np.rot90(temp_image_state_[i])
        #     # add non image and action conversion (up -> right, down -> left, left -> up, right -> down)
        #     tuples.append((np.copy(temp_image_state), non_image_state, action, reward, np.copy(temp_image_state_), non_image_state_, done))

        # for i in tuples:
        if (self.lidar or self.guide) and "image" in self.encoding:
            self.memory.store_transition(image_state, action, reward, image_state_, done, self.gamma, non_image_state, non_image_state_)
            # self.memory.add(image_state, action, reward, image_state_, done, non_image_state, non_image_state_)
        else:
            self.memory.store_transition(image_state, action, reward, image_state_, done, self.gamma)
            # self.memory.add(image_state, action, reward, image_state_, done)

    def sample_memory(self, beta, prioritized=False):
        if not prioritized:
            if (self.lidar or self.guide) and "image" in self.encoding:
                image_state, non_image_state, action, reward, new_image_state, new_non_image_state, done = \
                                        self.memory.sample_buffer(self.batch_size)
            else:
                image_state, action, reward, new_image_state, done = \
                                        self.memory.sample_buffer(self.batch_size)
            
            image_states = T.tensor(image_state).to(self.q_eval.device)
            
            rewards = T.tensor(reward).to(self.q_eval.device)
            dones = T.tensor(done).to(self.q_eval.device)
            actions = T.tensor(action).to(self.q_eval.device)
            image_states_ = T.tensor(new_image_state).to(self.q_eval.device)
            

            if (self.lidar or self.guide) and "image" in self.encoding:
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

    def learn(self, step):
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

            if (self.lidar or self.guide) and "image" in self.encoding:
                non_image_states=T.tensor(np.copy(data[:]['non_image_state'])).to(self.q_eval.device)
                non_image_states_=T.tensor(np.copy(data[:]['next_non_image_state'])).to(self.q_eval.device)
                # non_image_states = non_image_states.reshape((-1, 1))
                # non_image_states_ = non_image_states_.reshape((-1, 1))

                q_pred = self.q_eval.forward(image_states, non_image_states)[indices, actions]
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
            if (self.lidar or self.guide) and "image" in self.encoding:
                image_states, non_image_states, actions, rewards, image_states_, non_image_states_, dones = self.sample_memory(self.beta)
                q_pred = self.q_eval.forward(image_states, non_image_states)[indices, actions]
                q_next = self.q_next.forward(image_states_, non_image_states_).max(dim=1)[0]
            else:
                image_states, actions, rewards, image_states_, dones = self.sample_memory(self.beta)
                q_pred = self.q_eval.forward(image_states)[indices, actions]
                q_next = self.q_next.forward(image_states_).max(dim=1)[0]

            q_next[dones] = 0.0
            if self.nstep: q_target = rewards + (self.gamma**self.nstep_N)*q_next
            else: q_target = rewards + self.gamma*q_next
            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        loss.backward()
        self.q_eval.optimizer.step()

        self.learn_step_counter += 1

        if step == 0: self.decrement_epsilon()

        if self.prioritized: self.memory.update_priorities(tree_idxs, errors)

        return loss.cpu().detach().numpy()
    
    def finish_nstep(self):
        if self.prioritized:
            self.memory.finish_nstep(self.gamma)
        else:
            self.memory.finish_nstep(self.gamma)


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
    def __init__(self, nr, encoding, lidar, guide, max_size, batch_size, nstep, nstep_N, replay_alpha):
        self.encoding = encoding
        self.lidar = lidar
        self.guide = guide
        self.batch_size=batch_size
        self.capacity = max_size
        self.replay_alpha = replay_alpha
        self.transitions = SegmentTree(self.capacity)
        self.t = 0
        self.nr = nr
        self.nstep = nstep
        self.nstep_N = nstep_N
        self.nstep_buffer = []

    def store_transition(self, image_state, action, reward, next_image_state, done, gamma, non_image_state=None, next_non_image_state=None):
        if (self.lidar or self.guide) and "image" in self.encoding:
            if self.nstep:
                self.nstep_buffer.append((image_state, non_image_state, action, reward, next_image_state, next_non_image_state, done))

                if(len(self.nstep_buffer)<self.nstep_N*self.nr):
                    return
                
                R = sum([self.nstep_buffer[i][3]*(gamma**i) for i in range(0,self.nstep_N*self.nr,self.nr)])
                image_state, non_image_state, action, _, next_image_state, next_non_image_state, done = self.nstep_buffer.pop(0)
                self.transitions.append((self.t, image_state, non_image_state, action, R, next_image_state, next_non_image_state, done), self.transitions.max)  # Store new transition with maximum priority
            else:
                self.transitions.append((self.t, image_state, non_image_state, action, reward, next_image_state, next_non_image_state, done), self.transitions.max)
        else:
            if self.nstep:
                self.nstep_buffer.append((image_state, action, reward, next_image_state, done))

                if(len(self.nstep_buffer)<self.nstep_N*self.nr):
                    return
                
                R = sum([self.nstep_buffer[i][3]*(gamma**i) for i in range(0,self.nstep_N*self.nr,self.nr)])
                image_state, action, _, next_image_state, done = self.nstep_buffer.pop(0)
                self.transitions.append((self.t, image_state, action, R, next_image_state, done), self.transitions.max)  # Store new transition with maximum priority
            else:
                self.transitions.append((self.t, image_state, action, reward, next_image_state, done), self.transitions.max)
        self.t = 0 if done else self.t + 1  # Start new episodes with t = 0

    def finish_nstep(self, gamma):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][3]*(gamma**i) for i in range(0, len(self.nstep_buffer), self.nr)])
            if self.guide:
                state, other_state, action, _, state_, other_state_, done = self.nstep_buffer.pop(0)
                self.transitions.append((self.t, state, other_state, action, R, state_, other_state_, done), self.transitions.max)
            else:
                state, action, _, state_, done = self.nstep_buffer.pop(0)
                self.transitions.append((self.t, state, action, R, state_, done), self.transitions.max)
            
            self.t = 0 if done else self.t + 1
    
    def sample(self, replay_beta, epsilon=0.001):
        capacity = self.capacity if self.transitions.full else self.transitions.index
        while True:
            p_total=self.transitions.total()
            samples = np.random.uniform(0, p_total, self.batch_size)
            probs, data_idxs, tree_idxs = self.transitions.find(samples)
            probs = probs + epsilon
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
