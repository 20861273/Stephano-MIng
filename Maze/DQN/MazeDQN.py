import math
import random
from turtle import color
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T 

from dqn_maze import MazeAI, Direction, Point

#is_ipython = 'inline' in matplotlib.get_backend()
#if is_ipython: from IPython import display

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_features=n_states, out_features=24)   
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=n_actions)

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    # Exponential decay
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)

class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) # explore      
        else:
            with torch.no_grad():
                print(policy_net(state).argmax(dim=1))
                return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(self.device) # exploit

class MazeEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = MazeAI()
        self.done = False
        self.current_state = None

    def reset(self, sim, episode):
        self.current_state = torch.tensor([self.env.reset(sim, episode)], device=self.device).float()
    
    def grid(self):
        return self.env.grid

    def num_states(self):
        return self.env.grid.shape[1] * self.env.grid.shape[0]

    def num_actions_available(self):
        return len(Direction)

    def take_action(self, action):   
        #print("take action", action.item(), self.env.pos)     
        new_state, reward, self.done, _ = self.env.step(action.item())
        #print(self.done)
        return torch.tensor([new_state], device=self.device).float(), torch.tensor([reward], device=self.device)
        #return torch.from_numpy(np.array([new_state], dtype=np.float32)).to(device), torch.tensor([reward], device=self.device)

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod        
    def get_next(target_net, next_states):                
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

def plot(rewards, steps, learning_rate, discount_rate, exploration_rate):
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    avg_reward = 0
    avg_steps = 0

    if len(rewards) != 0: avg_reward = sum(rewards)/len(rewards)
    if len(steps) != 0: avg_steps = sum(steps)/len(steps)

    plt_title = "Results: α=%s, γ=%s, ϵ=%s\nAverage reward: %s\nAverage steps: %s" %(
        str(learning_rate), 
        str(discount_rate), 
        str(exploration_rate), 
        str(avg_reward), 
        str(avg_steps)
        )
    fig.suptitle(plt_title)

    ax[0].plot(np.arange(0, len(rewards)), rewards)
    ax[0].set_title('Rewards per episode')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Rewards')

    ax[1].plot(np.arange(0, len(steps)), steps)
    ax[1].set_title('Steps per episode')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('#Steps')

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.stack(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.stack(batch.next_state)

    return (t1,t2,t3,t4)


batch_size = 256
learning_rate = np.array([0.1])
discount_rate = np.array([0.98])
exploration_rate = np.array([1], dtype=np.float32)
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 10000
num_episodes = 10

steps_per_episode = []

sim = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for lr_i in np.arange(len(learning_rate)):
    for dr_i in np.arange(len(discount_rate)):
        for er_i in np.arange(len(exploration_rate)):

            em = MazeEnvManager(device)
            strategy = EpsilonGreedyStrategy(exploration_rate, eps_end, eps_decay)

            n_states = em.num_states()
            n_actions = em.num_actions_available()

            agent = Agent(strategy, n_actions, device)
            memory = ReplayMemory(memory_size)

            policy_net = DQN(n_states, n_actions).to(device)
            target_net = DQN(n_states, n_actions).to(device)

            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()

            optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate[lr_i])

            episode_durations = []
            rewards_per_episode = []

            for episode in range(num_episodes):
                print("Episode: ", episode)
                state = em.reset(sim, episode)

                for timestep in count():
                    #print(em.grid())
                    action = agent.select_action(state, policy_net)
                    next_state, reward = em.take_action(action)
                    #print("State:", state, "Action: ", action.item())
                    memory.push(Experience(state, action, next_state, reward))
                    state = next_state

                    if memory.can_provide_sample(batch_size):
                        experiences = memory.sample(batch_size)
                        states, actions, rewards, next_states = extract_tensors(experiences)

                        current_q_values = QValues.get_current(policy_net, states, actions)
                        next_q_values = QValues.get_next(target_net, next_states)
                        target_q_values = (next_q_values * discount_rate[dr_i]) + rewards

                        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if em.done:
                        episode_durations.append(timestep)
                        rewards_per_episode.append(reward.item())
                        steps_per_episode.append(timestep+1)
                        plot(rewards_per_episode, steps_per_episode, learning_rate[lr_i], discount_rate[dr_i], exploration_rate[er_i])
                        break

                if episode % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
            
            sim += 1

plt.show()
