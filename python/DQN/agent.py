import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import memory


class Qnet(nn.Module):
    def __init__(self, obs_dim, act_n):
        super(Qnet, self).__init__()
        self.obs_dim = obs_dim
        self.act_n = act_n
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN_Agent:
    def __init__(self, obs_dim, act_n, batch_size, lr=0.0001, gamma=0.9, e_greedy=0.1) -> None:
        self.obs_dim = obs_dim
        self.act_n = act_n
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.e_greedy = e_greedy
        self.Q = Qnet(obs_dim, act_n)
        self.Q_target = Qnet(obs_dim, act_n)
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(param.data)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.memory = memory.ReplyBuffer(100000)

    def sample(self, state, greedy=True):
        e_greedy = self.e_greedy if greedy else 0
        if np.random.random() < 1 - e_greedy:
            with torch.no_grad():
                state = torch.tensor(np.array([state]), dtype=torch.float32)
                q_values = self.Q(state)
                return q_values.max(1)[1].item()
        else:
            return np.random.randint(self.act_n)

    def store(self, exp):
        self.memory.store(exp)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self._process(
            self.memory.sample(self.batch_size))
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32)
        action_batch = torch.tensor(
            action_batch, dtype=torch.int64).unsqueeze(1)  # 添加维度
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(
            np.array(next_state_batch), dtype=torch.float32)
        done_batch = torch.tensor(np.float32(done_batch), dtype=torch.float32)
        q_values = self.Q(state_batch).gather(dim=1, index=action_batch)
        target_q_values = self.Q_target(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + \
            self.gamma * target_q_values * (1-done_batch)
        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    @staticmethod
    def _process(exps):
        n = len(exps)
        ret = []
        for i in range(5):
            ret.append([])
            for j in range(n):
                ret[i].append(exps[j][i])
        return ret
