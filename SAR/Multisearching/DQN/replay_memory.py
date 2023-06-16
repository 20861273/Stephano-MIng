import numpy as np
import torch as T

# last at sample buffer

class ReplayBuffer(object):
    def __init__(self, lidar, max_size, input_shape, n_actions):
        self.lidar = lidar # TODO
        self.mem_size = max_size
        self.mem_cntr = 0 # mem_cntr of the last stored memory
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

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

class PrioritizedReplayMemory(object):
    # Algorithm 1: Double DQN with proportional prioritization
    # 1: Input: minibatch k, step-size η, replay period K and size N, exponents α and β, budget T.
    # 2: Initialize replay memory H=∅, ∆=0, p1=1
    # 3: Observe S0 and choose A0 ∼ πθ(S0)
    # 4: for t=1 to T do
    # 5:    Observe St, Rt, γt
    # 6:    Store transition (St−1,At−1,Rt,γt,St) in H with maximal priority pt=max(i<t) pi
    # 7:    if t≡0 mod K then
    # 8:        for j=1 to k do
    # 9:            Sample transitionj∼P(j)=pαj / sumi(pαi)
    # 10:           Compute importance-sampling weight wj=(N·P(j))−β/maxi wi
    # 11:           Compute TD-error δj=Rj + γj * Qtarget(Sj,argmaxaQ(Sj,a)) − Q(Sj−1,Aj−1)
    # 12:           Update transition priority pj←|δj|
    # 13:           Accumulate weight-change ∆←∆+wj·δj·∇θQ(Sj−1,Aj−1)
    # 14:       endfor
    # 15:       Update weights θ←θ+η·∆, reset ∆=0
    # 16:       From time to time copy weights into target network θ target←θ
    # 17:   endif
    # 18: Choose action At∼πθ(St)
    # 19: endfor

    def __init__(self, max_mem, input_dims, n_actions, eps=0.00001, prob_alpha=0.5):
        self.prob_alpha = prob_alpha
        self.mem_size = max_mem
        self.memory = []
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

        self.priorities = np.zeros((self.mem_size,), dtype=np.float64) # has to be float 64, because float 32 had missing information
        self.max_priority = eps
        self.small_eps = eps
        self.alpha = prob_alpha

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        
        if index != 0:
            self.max_priority = max(self.priorities)
        else:
            self.max_priority = self.small_eps

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.priorities[index] = self.max_priority

        self.mem_cntr += 1

    def sample_buffer(self, gamma, batch_size, local, target, beta):
        # if buffer contains less samples than batch size
        max_mem = min(self.mem_cntr, self.mem_size)
        
        # Calculate probability of sampling transition
        prob_sum = np.sum(self.priorities)
        p = [priority / prob_sum for priority in self.priorities]
        # p[-1] = 1 - np.sum(p[0:-1]) #normalize

        batch = np.random.choice(max_mem, batch_size, p=p[:max_mem])

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]

        transitions_p = [p[idx] for idx in batch]

        # Compute importance-sampling weights
        weights = [pow(self.mem_size * p_j, -beta) for p_j in transitions_p]
        weights = list(weights)

        weights = weights / max(weights)

        td_error = self.get_td_error(batch_size, gamma, local, target, states, states_, actions, rewards, terminals)

        for td_error_idx, idx in enumerate(batch):
            self.priorities[idx] = pow(abs(td_error[td_error_idx]) + self.small_eps, self.prob_alpha).item()
            # print(pow(abs(td_error[td_error_idx]) + small_epsilon, alpha).item())

        return states, actions, rewards, states_, terminals, weights
        
    def get_td_error(self, batch_size, gamma, online_net, target_net, state, next_state, action, reward, done):
        state = T.stack(tuple(T.tensor(state).to(online_net.device)))
        next_state = T.stack(tuple(T.tensor(next_state).to(online_net.device)))
        action = T.tensor(action, dtype=T.int64).to(online_net.device)
        reward = T.tensor(reward, dtype=T.float32).to(online_net.device)
        done = T.tensor(done, dtype=T.bool).to(online_net.device)
        
        indices = np.arange(batch_size)
        pred = online_net.forward(state)[indices, action]
        next_pred = target_net.forward(next_state).max(dim=1)[0]

        pred = T.sum(pred.mul(action))

        target = reward + done * gamma * next_pred

        td_error = pred - target.detach()

        return td_error

    def __len__(self):
        return len(self.memory)
    
