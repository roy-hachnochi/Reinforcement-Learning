from Players.Player import Player
import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

TARGET_UPDATE = 5000
MEM_SIZE = 100000
BATCH_SIZE = 32
EPS_START = 1.0
EPS_END = 0.1

class DQPlayer(Player):
    def __init__(self, name, args, isLearning=False):
        super().__init__(name)
        self.name = name
        self.lr = args.lr  # learning rate
        self.eps_decay = args.eps_decay  # exploration rate decay
        self.gamma = args.gamma  # decay rate

        self.batch_size = args.batch_size  # batch size for optimization step
        self.target_update = args.target_update  # training step period for target network update

        self.isLearning = isLearning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policyNet = DQN(args).to(self.device)
        if self.isLearning:
            self.previousMove = {}
            self.memory = ReplayMemory(args.mem_size)
            self.targetNet = DQN(args).to(self.device)
            self.targetNet.load_state_dict(self.policyNet.state_dict())
            self.targetNet.eval()
            self.optimizer = optim.Adam(self.policyNet.parameters(), lr=self.lr)
            self.lossFunction = F.mse_loss
            self.nLearns = 0

    def get_action(self, state):
        board = torch.tensor(state.get_board(), device=self.device, dtype=torch.float).unsqueeze(0)
        actions = state.get_all_actions()
        illegal_inds = torch.tensor(state.get_illegal_actions(), device=self.device, dtype=torch.bool).unsqueeze(0)
        eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1 * self.nLearns / self.eps_decay) if self.isLearning else 0.0
        if random.random() < eps:  # explore
            legal_inds = [i for i in range(len(actions)) if not illegal_inds[0][i]]
            action_ind = random.choice(legal_inds)
        else:  # exploit
            qvals = self.get_legal_q_values(board, self.policyNet, illegal_inds).cpu().numpy().reshape(-1)
            action_ind = qvals.argmax()
        action = actions[action_ind]
        if self.isLearning:
            if self.previousMove.get(state.curPlayer) is not None:
                prev_board, prev_action = self.previousMove[state.curPlayer]
                reward = torch.zeros(1, 1, device=self.device)
                self.memory.push(prev_board, prev_action, board, reward, illegal_inds)
            action_ind = torch.tensor([[action_ind]], device=self.device)
            self.previousMove[state.curPlayer] = (board, action_ind)
        return action

    def save_policy(self, fileName):
        torch.save(self.policyNet.state_dict(), fileName)

    def load_policy(self, fileName):
        self.policyNet.load_state_dict(torch.load(fileName, map_location=self.device))
        if self.isLearning:
            self.targetNet.load_state_dict(self.policyNet.state_dict())

    def learn(self, reward, player):
        # push final state with reward:
        if self.isLearning:
            prev_board, prev_action = self.previousMove[player]
            reward = torch.tensor([[reward]], device=self.device, dtype=torch.float)
            self.memory.push(prev_board, prev_action, None, reward, None)

        # learn batch:
        if self.isLearning and len(self.memory) >= self.batch_size:
            # get replay memory batch:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                          device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            illegal_actions = torch.cat([s for s in batch.next_illegal_actions if s is not None])

            # get predicted q values:
            q_values = self.policyNet(state_batch).gather(1, action_batch)

            # get expected q values (with double deep q-learning):
            with torch.no_grad():
                next_state_q = self.get_legal_q_values(non_final_next_states, self.policyNet, illegal_actions)
                next_state_actions = torch.argmax(next_state_q, dim=1, keepdim=True)
                next_q_values = self.targetNet(non_final_next_states).gather(1, next_state_actions)
                next_state_values = torch.zeros(self.batch_size, 1, device=self.device)
                next_state_values[non_final_mask] = next_q_values.detach()
            expected_state_action_values = reward_batch + (self.gamma * next_state_values)

            # learn:
            loss = self.lossFunction(q_values, expected_state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policyNet.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            self.nLearns += 1
            if self.nLearns % self.target_update == 0:
                self.targetNet.load_state_dict(self.policyNet.state_dict())

    def get_q_values(self, board, model):
        return model(board.to(self.device))

    def get_legal_q_values(self, board, model, illegal_inds):
        with torch.no_grad():
            qvals = self.get_q_values(board, model).clone().detach()
        qvals[illegal_inds] = -np.inf  # to prevent from choosing illegal actions
        return qvals

    def stop_learning(self):
        self.isLearning = False

    def reset(self):
        if self.isLearning:
            self.previousMove = {}
        pass

# =================================================================================================================== #
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_illegal_actions'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# =================================================================================================================== #
class DQNArgs:
    def __init__(self, ch, h, w, output_size, layer_channels, layer_sizes, layer_strides, layer_padding,
                 batch_size, mem_size, target_update, eps_decay, lr, gamma):
        self.ch = ch  # number of input channels
        self.h = h  # input layer height
        self.w = w  # input layer width
        self.output_size = output_size  # output size
        self.depth = len(layer_channels)  # depth of network
        self.layer_channels = layer_channels  # number of channels for each layer
        self.layer_sizes = layer_sizes  # kernel sizes for each layer
        self.layer_strides = layer_strides  # stride for each layer
        self.layer_padding = layer_padding  # padding for each layer

        self.batch_size = batch_size  # batch size for optimization step
        self.mem_size = mem_size  # replay memory size
        self.target_update = target_update  # training step period for target network update

        self.eps_decay = eps_decay  # exploration rate decay
        self.lr = lr  # learning rate
        self.gamma = gamma  # MDP decay rate

    def conv2d_size_out(self, input_size, kernel_size, stride, padding):
        return (input_size + 2 * padding - kernel_size) // stride + 1

    def out_sizes(self):
        h = [self.h]
        w = [self.w]
        for l in range(self.depth):
            h.append(self.conv2d_size_out(h[l], self.layer_sizes[l], self.layer_strides[l], self.layer_padding[l]))
            w.append(self.conv2d_size_out(w[l], self.layer_sizes[l], self.layer_strides[l], self.layer_padding[l]))
        total_out_size = h[-1] * w[-1] * self.layer_channels[-1]
        return h, w, total_out_size


class DQN(nn.Module):
    def __init__(self, args):
        super(DQN, self).__init__()
        hl, wl, linear_input_size = args.out_sizes()

        self.convNet = nn.Sequential()
        channels = [args.ch] + args.layer_channels
        for l in range(args.depth):
            self.convNet.add_module('conv{}'.format(l),
                                    nn.Conv2d(channels[l], channels[l + 1],
                                              kernel_size=args.layer_sizes[l],
                                              stride=args.layer_strides[l],
                                              padding=args.layer_padding[l]))
            self.convNet.add_module('relu{}'.format(l), nn.ReLU())
            if (hl[l + 1] != 1) or (wl[l + 1] != 1):  # batchnorm only for nonscalar layers
                self.convNet.add_module('bn{}'.format(l), nn.BatchNorm2d(channels[l + 1]))

        self.fc_value = nn.Linear(linear_input_size, 1)
        self.fc_advantage = nn.Linear(linear_input_size, args.output_size)

    def forward(self, x):
        x = self.convNet(x)
        value = self.fc_value(x.view(x.size(0), -1))
        advantage = self.fc_advantage(x.view(x.size(0), -1))
        return value + advantage - torch.mean(advantage, 1, True)

# =================================================================================================================== #

if __name__ == '__main__':
    print("Deep Q-learning player")
