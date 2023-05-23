import numpy as np
import torch as T
from deep_q_network_simultaneous import DeepQNetwork
from replay_memory import ReplayBuffer, PrioritizedReplayMemory

class DQNAgent(object):
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

        if self.prioritized:
            self.memory = PrioritizedReplayMemory(mem_size, input_dims, n_actions, eps=0.0001, prob_alpha=0.5)
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
        if np.random.random() > self.epsilon:
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

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return      

        if self.prioritized:
            self.beta += 0.00005
            self.beta = min(1, self.beta)
            states, actions, rewards, states_, dones, weights = self.sample_memory(self.beta, self.prioritized)
                                                #batch_size, gamma, online_net, target_net, state, next_state, action, reward, done
            td_error = self.memory.get_td_error(self.batch_size, self.gamma, self.q_eval, self.q_next, states, states_, actions, rewards, dones)
            loss = pow(td_error, 2) * weights
            loss = loss.mean()
        else:
            states, actions, rewards, states_, dones = self.sample_memory()

            indices = np.arange(self.batch_size)
            q_pred = self.q_eval.forward(states)[indices, actions]
            q_next = self.q_next.forward(states_).max(dim=1)[0]

            q_next[dones] = 0.0
            q_target = rewards + self.gamma*q_next
            loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.replace_target_network()

        self.learn_step_counter += 1

        self.decrement_epsilon()