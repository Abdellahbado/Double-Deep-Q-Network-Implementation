from collections import namedtuple, deque
import random
from torch import nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 128       
GAMMA = 0.99
TAU = 1e-3
LR = 1e-4             
UPDATE_EVERY = 4

class DDQN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(DDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)  
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)
        self.qnetwork_local = DDQN(state_size, action_size, seed).to(device)
        self.qnetwork_target = DDQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=LR)  
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0
        self.eps = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
        self.episode_count = 0

    def step(self, state, action, reward, next_state, done, truncated):
        modified_reward = self._modify_reward(reward, done, truncated)
        self.memory.add(state, action, modified_reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def _modify_reward(self, reward, done, truncated):
        if done and not truncated:
            return -10.0
        elif truncated:
            return 10.0
        return 1.0  

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > self.eps:
            return action_values.argmax().item()
        else:
            return random.choice(np.arange(self.action_size))

    def end_episode(self):
        self.episode_count += 1
        if self.episode_count > 100:
            self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)