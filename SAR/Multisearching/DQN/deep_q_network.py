import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    """
    A class used to represent the environment

    ...

    Attributes
    ----------
    nr : int
        number of drones
    encoding : string
        keeps track of input encoding used
    n_actions : int
        number of actions in action space
    input_dims : tuple
        shape of input
    guide : boolean
        enables or disables closest unexplored cell state
    lidar : boolean
        enables or disables LiDAR state of surrounding cells
    c_dims : list
        number of channels for each convolutional layer
    k_size : list
        kernel size for each convolutional layer
    s_size : list
        stride size for each convolutional layer
    fc_dims : list
        number of neurons for each fully connected layer
    device_num : int
        which device (GPU/CPU) should be used
    name : string
        name of neural network, ie. DQN, DDQN
    chkpt_dir : string
        directory of checkpoint

    Methods
    -------
    calculate_conv_output_dims()
        calculation of fully connected layer input size
    forward(image_state, non_image_state=None)
        forward pass
    save_checkpoint(session, epoch, episode, time, loss)
        saves trained agent policy
    load_checkpoint()
        loads trained agent policy
    
    """
    def __init__(self, encoding, nr, lr,
                 n_actions, input_dims, guide, lidar, lstm, c_dims, k_size, s_size,
                    fc_dims, device_num, name, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.nr = nr

        self.encoding = encoding

        self.input_dims = input_dims

        self.lidar = lidar
        self.guide = guide
        self.lstm = lstm

        # self.lstm = lstm

        # self.lstm_h = lstm_h

        self.fc_dims = fc_dims.copy()
        self.n_actions = n_actions

        if "image" in self.encoding:
            self.c_dims = c_dims.copy()
            self.k_size = k_size.copy()
            self.s_size = s_size.copy()

            # convolutional layers
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
            self.conv1 = nn.Conv2d(input_dims[0], c_dims[0], k_size[0], stride=s_size[0])
            self.conv2 = nn.Conv2d(c_dims[0], c_dims[1], k_size[1], stride=s_size[1])
            # self.conv3 = nn.Conv2d(c_dims[1], c_dims[2], k_size[2], stride=s_size[2])

            # calculation of fully connected layer input size
            fc_input_dims = self.calculate_conv_output_dims()

            if self.lstm:
                self.lstm = nn.LSTM(fc_input_dims, self.fc_dims[0])
                self.fc1 = nn.Linear(fc_dims[0], self.n_actions)
            else:
                # fully connected layers
                self.fc1 = nn.Linear(fc_input_dims, fc_dims[0])
                
                # self.fc1 = nn.Linear(fc_input_dims, fc_dims[0])
                # self.fc2 = nn.Linear(fc_dims[0], fc_dims[1])


                # output layer
                self.fc2 = nn.Linear(fc_dims[0], self.n_actions)
                # self.fc3 = nn.Linear(fc_dims[1], self.n_actions)

            # random initialization of weights
            nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
            nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
            
            if self.lstm:
                for name, param in self.lstm.named_parameters():
                    if 'weight' in name:
                        nn.init.normal_(param, mean=0, std=0.01)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)  # Initialize biases to zero
                nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
            else:
                nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
                nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
                # nn.init.normal_(self.fc3.weight, mean=0, std=0.01)
        else:
            self.fc1 = nn.Linear(self.input_dims, self.fc_dims[0]) # * unpacking the input list
            self.fc2 = nn.Linear(self.fc_dims[0], self.fc_dims[1])
            # self.fc3 = nn.Linear(self.fc_dims[1], self.fc_dims[2])
            self.fc3 = nn.Linear(self.fc_dims[1], self.n_actions)
            
            nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
            nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
            nn.init.normal_(self.fc3.weight, mean=0, std=0.01)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device_num = device_num
        cuda_string = 'cuda:' + str(self.device_num)
        self.device = T.device(cuda_string if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self):
        state = T.zeros(1, *self.input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        # dims = self.conv3(dims)
        if self.lidar and not self.guide:
            return int(np.prod(dims.size())) + 4*self.nr #+ 1 # image size + surrounding states + percentage explored
        elif self.guide and not self.lidar:
            return int(np.prod(dims.size())) + 2
        elif self.lidar and self.guide:
            return int(np.prod(dims.size())) + 6
        else:
            return int(np.prod(dims.size())) # image size

    def forward(self, image_state, non_image_state=None):
        # print(image_state.device)
        if "image" in self.encoding:
            # forward pass convolutional layers
            conv1 = F.relu(self.conv1(image_state))
            # conv1 = self.maxpool(conv1)
            conv2 = F.relu(self.conv2(conv1))
            # conv2 = self.maxpool(conv2)
            # conv3 = F.relu(self.conv3(conv2))
            # conv3 shape is BS x n_filters x H x W

            # reshape with view, basically np.reshape ( conv_state shape is BS x (n_filters * H * W) )
            # conv_state = conv1.view(conv1.size()[0], -1)
            conv_state = conv2.view(conv2.size()[0], -1)
            # conv_state = conv3.view(conv3.size()[0], -1)
            

            if self.lidar or self.guide:
                # consentrate output of convolutional layers and non-image state
                concatenated_state = T.cat((conv_state, non_image_state), dim=1)

                if self.lstm:
                    concatenated_state = concatenated_state.unsqueeze(0)
                    lstm_out, self.hidden = self.lstm(concatenated_state, self.hidden)
                else:
                    # forward pass fully connected layers
                    flat1 = F.relu(self.fc1(concatenated_state))
                    # flat2 = F.relu(self.fc2(flat1))
            else:
                flat1 = F.relu(self.fc1(conv_state))
            if self.lstm:
                actions = F.relu(self.fc1(lstm_out.view(-1, self.fc_dims[0])))
            else:
                actions = self.fc2(flat1)
                # actions = self.fc3(flat2)
        else:
            # image_state = image_state.to(self.device)
            x = F.relu(self.fc1(image_state))
            x = F.relu(self.fc2(x))
            # x = F.relu(self.fc3(x))
            actions = self.fc3(x)

        return actions

    def init_hidden(self, batch_size):
        if self.training:
            self.hidden = (T.zeros(1, batch_size, self.fc_dims[0]).to(self.device),
                        T.zeros(1, batch_size, self.fc_dims[0]).to(self.device))
        else:
            self.hidden = (T.zeros(1, 1, self.fc_dims[0]).to(self.device),
                        T.zeros(1, 1, self.fc_dims[0]).to(self.device))

    # saves trained agent policy
    def save_checkpoint(self, session, epoch, episode, time, loss):
        T.save({
            'session': session,
            'epoch': epoch,
            'episode': episode,
            'time': time,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, self.checkpoint_file)

    # loads trained agent policy
    def load_checkpoint(self):
        print('... loading checkpoint ...')
        cuda_string = 'cuda:' + str(self.device_num)
        checkpoint = T.load(self.checkpoint_file, map_location=cuda_string)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, T.Tensor):
                    state[k] = v.cuda(device=self.device_num)
        return checkpoint