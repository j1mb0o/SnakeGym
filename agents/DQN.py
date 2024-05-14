import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

import gym
from gym import spaces

# Pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

import random
from collections import deque



class Net(nn.Module):
    def __init__(self, state_size, action_size, learning_rate):
        super(Net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.fc1 = nn.Linear(self.state_size, 84)
        self.fc2 = nn.Linear(84, 84)
        self.fc3 = nn.Linear(84, self.action_size)
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_step(self, state, target):
        self.optimizer.zero_grad()
        pred = self.forward(state)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class DQNAgent:
    def __init__(self, state_size, action_size, test=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000000)
        self.gamma = 0.99   
        self.epsilon = 1.0 if not test else 0.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = Net(self.state_size, self.action_size, self.learning_rate).to(self.device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            act_values = self.model(state)
            return np.argmax(act_values.cpu().detach().numpy()[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
                target = (reward + self.gamma *
                          np.amax(self.model(next_state).cpu().detach().numpy()[0]))
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            target_f = self.model(state)
            target_f[0][action] = target
            self.model.train_step(state, target_f)

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


if __name__ == "__main__":
    agent = DQNAgent(25, 4)