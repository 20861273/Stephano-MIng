import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, nr, lr, n_actions, input_dims, fc1_dims, fc2_dims, fc3_dims, name, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions

        # self.conv1 = nn.Conv2d(input_dims[0], 16, 4, stride=1)
        self.conv1 = nn.Conv2d(input_dims[0], 16, 2, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 2, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims()

        self.fc1 = nn.Linear(fc_input_dims, 64)  # 5*5 from image dimension
        self.fc2 = nn.Linear(64, n_actions)

        # self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims) # * unpacking the input list
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        # self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self):
        state = T.zeros(1, *self.input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        # dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        # conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W

        # conv_state = conv1.view(conv1.size()[0], -1)
        conv_state = conv2.view(conv2.size()[0], -1)
        # conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # actions = self.fc4(x)

        return actions

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))