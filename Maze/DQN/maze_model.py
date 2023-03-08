import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Agent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Agent, self).__init__()
        self.hidden_size = 128
        self.fc1 = nn.Linear(state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x