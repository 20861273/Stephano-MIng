import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from maze_model import Agent
from maze_environment import Environment

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class Trainer():
    def __init__(self, env, agent, optimizer, replay_buffer, target_update=10, batch_size=32, gamma=0.99):
        self.env = env
        self.agent = agent
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.target_update = target_update
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_agent = Agent(env.observation_space.shape[0], env.action_space.n, hidden_size=128)
        self.target_agent.load_state_dict(agent.state_dict())

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                epsilon = max(0.01, 1 - episode / 500)
                action = self.act(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                self.replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                self.update()

            if episode % self.target_update == 0:
                self.target_agent.load_state_dict(self.agent.state_dict())

            print("Episode: {}, Total Reward: {}".format(episode, total_reward))

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.agent(state)
                action = q_values.argmax().item()
                return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        q_values = self.agent(state).gather(1, action)
        next_q_values = self.target_agent(next_state).max(1)[0].unsqueeze(1)
        expected_q_values = reward + (1 - done) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, expected_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# set up environment and agent
env = Environment(grid_size=10)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, hidden_size=128)

# set up training parameters
num_episodes = 1000
max_steps = 100
batch_size = 64
gamma = 0.99
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995
buffer_size = 10000

# set up replay buffer
replay_buffer = ReplayBuffer(buffer_size)

# set up target network
target_agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, hidden_size=128)
target_agent.load_state_dict(agent.state_dict())
target_agent.eval()

# set up optimizer
optimizer = optim.Adam(agent.parameters(), lr=0.001)

# training loop
for episode in range(num_episodes):
    state = env.reset()
    eps = eps_end + (eps_start - eps_end) * np.exp(-episode * eps_decay)
    for step in range(max_steps):
        # choose action based on epsilon-greedy policy
        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = agent(state_tensor)
            action = q_values.detach().numpy().argmax()
        # execute action in environment
        next_state, reward, done = env.step(action)
        # add experience to replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        # train agent
        if len(replay_buffer) > batch_size:
            agent.train(replay_buffer, batch_size)
        # update target network
        if step % 10 == 0:
            target_agent.load_state_dict(agent.state_dict())
        # end episode if done
        if done:
            break

