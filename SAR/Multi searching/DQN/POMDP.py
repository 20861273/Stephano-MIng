import torch
import torch.nn as nn
import torch.optim as optim

from dqn_environment import Environment, HEIGHT, WIDTH

# Define the neural network:
class QNetwork(nn.Module):
    def __init__(self, in_features, hidden_size, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define POMDP
def update_belief_state(self, belief_state, observation, transition_matrix, observation_matrix, action):
    # update belief state
    new_belief_state = observation_matrix[:,observation] * transition_matrix[action].dot(belief_state)
    
    # normalize
    new_belief_state = new_belief_state / new_belief_state.sum()
    
    return new_belief_state

# Initialize variables
num_episodes = 1000
env = Environment()
initial_belief_state = [0.01]*WIDTH*HEIGHT

# Define Q-learning
q_network = QNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(q_network.parameters())

for episode in range(num_episodes):
    # reset the environment and get the initial state
    state = env.reset()
    while True:
        # select an action
        with torch.no_grad():
            q_values = q_network(state)
            action = torch.argmax(q_values).item()
        
        # take the action and get the next state, reward, and done
        next_state, reward, done, _ = env.step(action)
        
        # calculate the target
        with torch.no_grad():
            target = reward + discount_factor * torch.max(q_network(next_state))
        
        # calculate the loss
        loss = criterion(q_values[action], target)
        
        # update the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update the state
        state = next_state
        
        # check if the episode is done
        if done:
            break

# Train the model
for episode in range(num_episodes):
    # reset the environment and get the initial state
    state = env.reset()
    belief_state = initial_belief_state
    while True:
        # select an action using the Q-network
        with torch.no_grad():
            q_values = q_network(state)
            action = torch.argmax(q_values).item()
        
        # take the action and get the next state, observation, reward, and done
        next_state, observation, reward, done, _ = env.step(action)
        
        # update the belief state
        belief_state = update_belief_state(belief_state, observation, transition_matrix, observation_matrix, action)
        
        # calculate the target
        with torch.no_grad():
            target = reward + discount_factor * torch.max(q_network(next_state))
        
        # calculate the loss
        loss = criterion(q_values[action], target)
        
        # update the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update the state
        state = next_state
        
        # check if the episode is done
        if done:
            break

    
