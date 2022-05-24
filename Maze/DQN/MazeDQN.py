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

from rl_maze import MazeAI, Direction, Point

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
                return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(self.device) # exploit

class MazeEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = MazeAI()
        self.done = False

    def reset(self):
        self.env.reset()
    
    def grid(self):
        return self.env.grid

    def num_actions_available(self):
        return len(Direction)

    def take_action(self, action):   
        #print("take action", action.item(), self.env.pos)     
        reward, self.done, _ = self.env.step(action.item())
        #print(self.done)
        return torch.tensor([reward], device=self.device)

    def get_state(self):
        #print("get state")
        pos = self.env.pos
        point_l = Point(pos.x - 1, pos.y)
        point_r = Point(pos.x + 1, pos.y)
        point_u = Point(pos.x, pos.y - 1)
        point_d = Point(pos.x, pos.y + 1)
        #print(self.env.direction, Direction.LEFT)
        dir_l = self.env.direction == Direction.LEFT
        dir_r = self.env.direction == Direction.RIGHT
        dir_u = self.env.direction == Direction.UP
        dir_d = self.env.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.env.is_collision(point_r)) or 
            (dir_l and self.env.is_collision(point_l)) or 
            (dir_u and self.env.is_collision(point_u)) or 
            (dir_d and self.env.is_collision(point_d)),

            # Danger right
            (dir_u and self.env.is_collision(point_r)) or 
            (dir_d and self.env.is_collision(point_l)) or 
            (dir_l and self.env.is_collision(point_u)) or 
            (dir_r and self.env.is_collision(point_d)),

            # Danger left
            (dir_d and self.env.is_collision(point_r)) or 
            (dir_u and self.env.is_collision(point_l)) or 
            (dir_r and self.env.is_collision(point_u)) or 
            (dir_l and self.env.is_collision(point_d)),

            # Danger back
            (dir_d and self.env.is_collision(point_u)) or 
            (dir_u and self.env.is_collision(point_d)) or 
            (dir_r and self.env.is_collision(point_l)) or 
            (dir_l and self.env.is_collision(point_r)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d
            ]

        return torch.from_numpy(np.array(state, dtype=np.float32)).to(device)

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

def plot(reward, ep_dur, moving_avg_period):

    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.plot(reward)
    
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(ep_dur)

    moving_avg = get_moving_average(moving_avg_period, ep_dur)
    plt.plot(moving_avg)    
    plt.pause(0.001)
    print("Episode", len(ep_dur), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])

    #if is_ipython: display.clear_output(wait=True)

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
gamma = 0.9
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 10000
lr = 0.01
num_episodes = 200



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

em = MazeEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

n_states = len(em.get_state())
n_actions = em.num_actions_available()

agent = Agent(strategy, n_actions, device)
memory = ReplayMemory(memory_size)

policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

episode_durations = []
rewards_per_episode = []

for episode in range(num_episodes):
    print("Episode: ", episode)
    em.reset()
    state = em.get_state()

    for timestep in count():
        print(em.grid())
        action = agent.select_action(state, policy_net)
        reward = em.take_action(action)
        next_state = em.get_state()
        print("State:", state, "Action: ", action.item())
        memory.push(Experience(state, action, next_state, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if em.done:
            episode_durations.append(timestep)
            rewards_per_episode.append(reward.item())
            plot(rewards_per_episode, episode_durations, 100)
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

plt.show()
