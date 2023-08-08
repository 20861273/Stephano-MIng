import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepRQNetwork(nn.Module):
    # def __init__(self, encoding, nr, lr,
    #              n_actions, input_dims, lidar, lstm, c_dims, k_size, s_size,
    #                 fc_dims, lstm_h, device_num, name, chkpt_dir):
    def __init__(self, encoding, nr, lr,
                 n_actions, input_dims, lidar, c_dims, k_size, s_size,
                    fc_dims, device_num, name, chkpt_dir):
        super(DeepRQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.nr = nr

        self.encoding = encoding

        self.input_dims = input_dims

        self.lidar = lidar

        # self.lstm = lstm

        # self.lstm_h = lstm_h

        self.fc_dims = fc_dims.copy()
        self.n_actions = n_actions

        if "image" in self.encoding:
            self.c_dims = c_dims.copy()
            self.k_size = k_size.copy()
            self.s_size = s_size.copy()

            self.conv1 = nn.Conv2d(self.input_dims[0], self.c_dims[0], self.k_size[0], stride=self.s_size[0])

            # nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(self.c_dims[0], self.c_dims[1], self.k_size[1], stride=self.s_size[1])
            # nn.MaxPool2d(kernel_size=2)
            # self.conv3 = nn.Conv2d(self.c_dims[1], self.c_dims[2], self.k_size[2], stride=self.s_size[2])

            fc_input_dims = self.calculate_conv_output_dims()

            self.lstm = nn.LSTM(fc_input_dims, self.fc_dims[0])
            self.fc1 = nn.Linear(fc_dims[0], self.n_actions)
        else:
            self.fc1 = nn.Linear(*self.input_dims, self.fc_dims[0]) # * unpacking the input list
            self.fc2 = nn.Linear(self.fc_dims[0], self.fc_dims[1])
            self.fc3 = nn.Linear(self.fc_dims[1], self.fc_dims[2])
            self.fc4 = nn.Linear(self.fc_dims[2], self.n_actions)
        
        # if self.lstm:
            
        #     self.lstm = nn.LSTM(self.n_actions, self.lstm_h)
        #     self.fc3 = nn.Linear(fc_dims[0], self.n_actions)

        
        
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
        return int(np.prod(dims.size())) + 1
        if self.lidar:
            return int(np.prod(dims.size())) + 4*self.nr #+ 1 # image size + surrounding states + percentage explored
        else:
            return int(np.prod(dims.size())) # image size

    def forward(self, image_state, non_image_state=None):
        # print(image_state.device)
        if "image" in self.encoding:
            conv1 = F.relu(self.conv1(image_state))
            conv2 = F.relu(self.conv2(conv1))
            # conv3 = F.relu(self.conv3(conv2))
            # conv3 shape is BS x n_filters x H x W

            # conv_state = conv1.view(conv1.size()[0], -1)
            conv_state = conv2.view(conv2.size()[0], -1)
            # conv_state = conv3.view(conv3.size()[0], -1)
            # conv_state shape is BS x (n_filters * H * W)
            concatenated_state = T.cat((conv_state, non_image_state), dim=1)
            concatenated_state = concatenated_state.unsqueeze(0)
            # concatenated_state = concatenated_state.transpose(0, 1)

            lstm_out, self.hidden = self.lstm(concatenated_state, self.hidden)
            actions = F.relu(self.fc1(lstm_out.view(-1, self.fc_dims[0])))
            # flat1 = flat1.squeeze(0)

            # if self.lidar:
            #     # consentrate output of convolutional layers and non-image state
            #     concatenated_state = T.cat((conv_state, non_image_state), dim=1)

            #     flat1 = F.relu(self.fc1(concatenated_state))
            # else:
            #     flat1 = F.relu(self.fc1(conv_state))
            # actions = self.fc1(flat1)

            return actions
        else:
            # image_state = image_state.to(self.device)
            x = F.relu(self.fc1(image_state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            actions = self.fc4(x)

        return actions
    
    def init_hidden(self, batch_size):
        if self.training:
            self.hidden = (T.zeros(1, batch_size, self.fc_dims[0]).to(self.device),
                        T.zeros(1, batch_size, self.fc_dims[0]).to(self.device))
        else:
            self.hidden = (T.zeros(1, 1, self.fc_dims[0]).to(self.device),
                        T.zeros(1, 1, self.fc_dims[0]).to(self.device))

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