import numpy as np
import random 
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

##Importing the model (function approximator for Q-table)
from model import QNetwork
from dqn_environment import Environment, HEIGHT, WIDTH, Point
from dqn_save_results import print_results

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  #replay buffer size
BATCH_SIZE = 64         # minibatch size
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns form environment."""
    
    def __init__(self, learning_rate, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        #Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=learning_rate)
        
        # Replay memory 
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE,BATCH_SIZE,seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_step, done, discount_rate, learning_rate):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory)>BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, discount_rate, learning_rate)
    def act(self, state, eps = 0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        action = np.array([0,0,0,0])
        if random.random() > eps:
            action[np.argmax(action_values.cpu().data.numpy())] = 1
            return action
            # return np.argmax(action_values.cpu().data.numpy())
        else:
            action[random.choice(np.arange(self.action_size))] = 1
            return action
            # return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences, gamma, tau):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1) # unsqueeze(1) converts the dimension from (batch_size) to (batch_size,1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = torch.zeros(rewards.size(0), predicted_targets.size(1), device=device) + rewards + (gamma * labels_next * (1 - dones))
        
        loss = criterion(predicted_targets,labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,tau)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            
class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def dqn(n_ts, n_episodes, max_t, eps_start, eps_max, eps_end, eps_decay, positive_reward, load_path, save_path):
    """Deep Q-Learning
    
    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon 
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
        
    """
    mode = True
    all_grids = []
    all_rewards = []
    all_steps = []
    scores = [] # list containing score from each episode
    scores_window = deque(maxlen=1000) # last 100 scores
    rewards_per_episode = []
    training_time = time.time()
    print("Training starting...")
    for ts_i in range(0, n_ts):
        print("Training session: ", ts_i)
        seq_rewards = []
        seq_steps = []
        sim = 0
        for lr_i in np.arange(len(learning_rate)):
            for dr_i in np.arange(len(discount_rate)):
                for er_i in np.arange(len(exploration_rate)):
                    print("\nTraining simulation: %s\nLearning rate = %s\nDiscount rate = %s\nExploration rate = %s\nExploration rate min = %s\nExploration rate max = %s\nExploration decay rate = %s"
                            %(sim, learning_rate[lr_i], discount_rate[dr_i], exploration_rate[er_i], min_exploration_rate[er_i], max_exploration_rate[er_i], exploration_decay_rate[er_i]))
                    agent = Agent(learning_rate[lr_i], state_size=HEIGHT*WIDTH,action_size=4,seed=0)
                    eps = eps_start[er_i]
                    env = Environment(positive_reward)
                    rewards_per_episode = []
                    steps_per_episode = [] 
                    for i_episode in range(0, n_episodes):
                        if i_episode % 1000 == 0 or i_episode == 0: print("Episode: ", i_episode)
                        state = env.reset()
                        score = 0
                        rewards_current_episode = 0
                        done = False
                        for t in range(max_t):
                            if i_episode == 0 and mode: all_grids.append(env.grid.copy())
                            action = agent.act(state,eps)
                            next_state,reward,done,_ = env.step(action)
                            agent.step(state,action,reward,next_state,done,discount_rate[dr_i],learning_rate[lr_i])

                            ## above step decides whether we will train(learn) the network
                            ## actor (local_qnetwork) or we will fill the replay buffer
                            ## if len replay buffer is equal to the batch size then we will
                            ## train the network or otherwise we will add experience tuple in our 
                            ## replay buffer.
                            state = next_state
                            score += reward
                            rewards_current_episode += reward
                            if done:
                                rewards_per_episode.append(rewards_current_episode)
                                steps_per_episode.append(t+1)
                                if i_episode == 0 and mode: all_grids.append(env.grid.copy())
                                break
                            scores_window.append(score) ## save the most recent score
                            scores.append(score) ## sae the most recent score
                            # eps = max(eps*eps_decay[er_i],eps_end[er_i])## decrease the epsilon
                            eps = eps_end[er_i] + \
                                        (eps_max[er_i] - eps_end[er_i]) * np.exp(-eps_decay[er_i]*i_episode)

                        if i_episode % 1000==0 and not i_episode == 0:
                            print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode,np.mean(scores_window)))
                        
                        if mode:
                            if not done:
                                rewards_per_episode.append(rewards_current_episode)
                                steps_per_episode.append(t+1)
                    
                    file_name = "experience%s.pth" %(str(sim))
                    file_name = os.path.join(save_path, file_name)
                    torch.save(agent.qnetwork_local.state_dict(),file_name)
                    if mode:
                        # Adds epoch rewards to training session rewards variable
                        # rewards_per_episode: (num_episodes, self.nr) ; i.e,  [[agent0_reward0, agent1_reward0], [agent0_reward1, agent1_reward1], ...]
                        # new_tmp_exp_rewards: (2*num_episodes,)
                        # new_exp_rewards: (num_sequences, self.nr, num_episodes)
                        tmp_seq_rewards = np.array(seq_rewards)
                        new_tmp_seq_rewards = np.array(np.append(tmp_seq_rewards.ravel(),np.array(rewards_per_episode)))
                        if tmp_seq_rewards.shape[0] == 0:
                            new_seq_rewards = new_tmp_seq_rewards.reshape(1,len(rewards_per_episode))
                        else:
                            new_seq_rewards = new_tmp_seq_rewards.reshape(tmp_seq_rewards.shape[0]+1,tmp_seq_rewards.shape[1])
                        seq_rewards = new_seq_rewards.tolist()

                        tmp_seq_steps = np.array(seq_steps)
                        new_tmp_seq_steps = np.array(np.append(tmp_seq_steps.ravel(),np.array(steps_per_episode)))
                        if tmp_seq_steps.shape[0] == 0:
                            new_seq_steps = new_tmp_seq_steps.reshape(1,len(steps_per_episode))
                        else:
                            new_seq_steps = new_tmp_seq_steps.reshape(tmp_seq_steps.shape[0]+1,tmp_seq_steps.shape[1])
                        seq_steps = new_seq_steps.tolist()
                    sim += 1
    
        tmp_rewards = np.array(all_rewards)
        new_tmp_rewards = np.array(np.append(tmp_rewards.ravel(),np.array(seq_rewards).ravel()))
        new_rewards = new_tmp_rewards.reshape(ts_i+1,sim,n_episodes)
        all_rewards = new_rewards.tolist()

        tmp_steps = np.array(all_steps)
        new_tmp_steps = np.array(np.append(tmp_steps.ravel(),np.array(seq_steps).ravel()))
        new_steps = new_tmp_steps.reshape(ts_i+1,sim,num_episodes)
        all_steps = new_steps.tolist()

    avg_rewards, avg_steps = calc_avg(new_rewards, new_steps, len(learning_rate)*len(discount_rate*len(exploration_rate)))
    training_time = time.time() - training_time
    print("Time to train policy: %sm %ss" %(divmod(training_time, 60)))

    results = print_results(env.grid, HEIGHT, WIDTH)

    results.plot(avg_rewards, avg_steps, learning_rate, discount_rate, exploration_rate, save_path, env, training_time, positive_reward)

def moving_avarage_smoothing(X,k):
	S = np.zeros(X.shape[0])
	for t in range(X.shape[0]):
		if t < k:
			S[t] = np.mean(X[:t+1])
		else:
			S[t] = np.sum(X[t-k:t])/k
	return S

# Calculates average rewards and steps
def calc_avg(rewards, steps, num_sims):
    avg_rewards = np.sum(np.array(rewards), axis=0)
    avg_steps = np.sum(np.array(steps), axis=0)

    avg_rewards = np.divide(avg_rewards, num_sims)
    avg_steps = np.divide(avg_steps, num_sims)

    mov_avg_rewards = np.empty(avg_rewards.shape)
    mov_avg_steps = np.empty(avg_steps.shape)

    for i in range(0, num_sims):
        mov_avg_rewards[i] = moving_avarage_smoothing(avg_rewards[i], 100)
        mov_avg_steps[i] = moving_avarage_smoothing(avg_steps[i], 100)

    return mov_avg_rewards.tolist(), mov_avg_steps.tolist()

# Initializing Q-Learning Parameters
num_episodes = 200000
max_steps_per_episode = 200
num_sims = 1

learning_rate = np.array([0.00075])
discount_rate = np.array([0.002])

exploration_rate = np.array([0.03], dtype=np.float32)
max_exploration_rate = np.array([0.03], dtype=np.float32)
min_exploration_rate = np.array([0.03], dtype=np.float32)
exploration_decay_rate = np.array([0.03], dtype=np.float32)

positive_reward = 100 # 20, 40 , (lr, dr)

PATH = os.getcwd()
PATH = os.path.join(PATH, 'SAR')
PATH = os.path.join(PATH, 'Results')
PATH = os.path.join(PATH, 'DQN')
load_path = os.path.join(PATH, 'Saved_data')
if not os.path.exists(load_path): os.makedirs(load_path)
date_and_time = datetime.now()
save_path = os.path.join(PATH, date_and_time.strftime("%d-%m-%Y %Hh%Mm%Ss"))
if not os.path.exists(save_path): os.makedirs(save_path)

print(save_path)

# On or off policy
policy_bool = True
policy_num = 2

if policy_bool:
    dqn(num_sims, num_episodes, max_steps_per_episode, exploration_rate, max_exploration_rate, min_exploration_rate, exploration_decay_rate, positive_reward, load_path, save_path)
else:
    agent = Agent(0, state_size=HEIGHT*WIDTH,action_size=4,seed=0)
    file_name = "experience%s.pth" %(str(policy_num))
    file_name = os.path.join(load_path, file_name)
    agent.qnetwork_local.load_state_dict(torch.load(file_name))
    agent.qnetwork_local.eval()
    env = Environment(positive_reward)

    found = []
    resultss = []
    cnt = 0
    for y in range(env.grid.shape[0]):
        for x in range(env.grid.shape[1]):
            state = env.reset()
            grids = []
            results = []
            
            env.grid[env.pos.y, env.pos.x] = 0
            env.pos = Point(x,y)
            env.prev_pos = Point(x,y)
            env.starting_pos = Point(x,y)
            env.grid[env.pos.y, env.pos.x] = 2

            grids.append(env.grid.copy())
            results.append((env.pos, None))

            done = False
           
            for step in range(max_steps_per_episode):
                action = agent.act(state,eps=0)
                next_state, reward, done, _ = env.step(action)
                results.append((env.pos, action))
                grids.append(env.grid.copy())
                state = next_state
                if done:
                    found.append(grids)
                    resultss.append(results)
                    cnt += 1
                    break
            # if done: break
        # if done: break
    print("Percentage success: ", cnt, HEIGHT*WIDTH, cnt/(HEIGHT*WIDTH)*100)
    cnt = 0
    for j, g in enumerate(found):
        for i, g_i in enumerate(g):
            PR = print_results(g_i, env.grid.shape[0], env.grid.shape[1])
            PR.print_graph(i)
            
            file_name = "plot-%s%s.png" %(j, i)
            plt.savefig(os.path.join(save_path, file_name))
            plt.close()

