import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, id, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims= fc3_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoints/" + name + '_ddpg' + '_fc1' + str(self.fc1_dims) + '_fc2' + str(self.fc2_dims) + '_fc3' + str(self.fc3_dims)  + '_' +str(id))
        print(self.checkpoint_file, flush=True)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)
        self.bn4 = nn.LayerNorm(self.fc2_dims)
        self.bn5 = nn.LayerNorm(self.fc3_dims)
        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        #the sizes need to match up
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims) #fc2_dims can be replaced
        self.action_value2 = nn.Linear(self.fc2_dims, self.fc3_dims)
        
        self.q = nn.Linear(self.fc3_dims, 1)
        self.q_bef = nn.Linear(self.fc3_dims, self.fc3_dims) #fc2_dims can be replaced

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)

        f4 = 0.003
        self.q.weight.data.uniform_(-f4, f4)
        self.q.bias.data.uniform_(-f4, f4)

        f5 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f5, f5)
        self.action_value.bias.data.uniform_(-f5, f5)

        f6 = 1./np.sqrt(self.action_value2.weight.data.size()[0])
        self.action_value2.weight.data.uniform_(-f6, f6)
        self.action_value2.bias.data.uniform_(-f6, f6)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = self.bn3(state_value)

        #state_value = F.relu(state_value)
        #action_value = F.relu(self.action_value(action))
        action_value = self.action_value(action)
        action_value = self.bn4(action_value)
        action_value = F.relu(action_value)
        action_value = self.action_value2(action_value)
        action_value = self.bn5(action_value)
        
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q_bef(state_action_value)
        state_action_value = F.tanh(state_action_value)
        state_action_value = self.q(state_action_value)


        return state_action_value

    def save_checkpoint(self, extension):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file + extension)

    def load_checkpoint(self, extension):
        print('... loading checkpoint ...')
        print("Loading checkpoint ", self.checkpoint_file + extension, flush=True)
        print(self.checkpoint_file, flush=True)
        self.load_state_dict(T.load(self.checkpoint_file + extension))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)

class ActorNetwork(nn.Module):
    def __init__(self, id, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims= fc3_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoints/" + name + '_ddpg' + '_fc1' + str(self.fc1_dims) + '_fc2' + str(self.fc2_dims) + '_fc3' + str(self.fc3_dims)  + '_' +str(id))

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.bn3 = nn.LayerNorm(self.fc3_dims)

        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.mu = nn.Linear(self.fc3_dims, self.n_actions)

        f3 = 1./np.sqrt(self.fc3.weight.data.size()[0])
        self.fc3.weight.data.uniform_(-f3, f3)
        self.fc3.bias.data.uniform_(-f3, f3)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f4 = 0.003
        self.mu.weight.data.uniform_(-f4, f4)
        self.mu.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self, extension):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file + extension)

    def load_checkpoint(self, extension):
        print('... loading checkpoint ...')
        print("Loading checkpoint ", self.checkpoint_file + extension, flush=True)
        print(self.checkpoint_file, flush=True)
        self.load_state_dict(T.load(self.checkpoint_file + extension))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
