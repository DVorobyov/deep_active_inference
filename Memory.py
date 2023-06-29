import torch
import numpy as np


class ReplayBuffer():
    def __init__(self, capacity, obs_shape, device='cpu'):
        self.device = device
        self.capacity = capacity
        self.obs_mem = torch.empty([capacity] + [dim for dim in obs_shape], dtype=torch.float32, device=self.device)
        self.action_mem = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.reward_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.done_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.push_count = 0

    def push(self, obs, action, reward, done):
        self.obs_mem[self.position()] = obs
        self.action_mem[self.position()] = action
        self.reward_mem[self.position()] = reward
        self.done_mem[self.position()] = done
        self.push_count += 1

    def position(self):
        return self.push_count % self.capacity

    def sample(self, obs_indices, action_indices, reward_indices, done_indices, max_n_indices, batch_size):
        end_indices = np.random.choice(min(self.push_count, self.capacity) - max_n_indices * 2, batch_size,
                                       replace=False) + max_n_indices
        for i in range(len(end_indices)):
            if end_indices[i] in range(self.position(), self.position() + max_n_indices):
                end_indices[i] += max_n_indices
        obs_batch = self.obs_mem[np.array([index - obs_indices for index in end_indices])]
        action_batch = self.action_mem[np.array([index - action_indices for index in end_indices])]
        reward_batch = self.reward_mem[np.array([index - reward_indices for index in end_indices])]
        done_batch = self.done_mem[np.array([index - done_indices for index in end_indices])]
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, max_n_indices):
                if self.done_mem[index - j]:
                    for k in range(len(obs_indices)):
                        if obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.obs_mem[0])
                    for k in range(len(action_indices)):
                        if action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.action_mem[0])
                    for k in range(len(reward_indices)):
                        if reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(self.reward_mem[0])
                    for k in range(len(done_indices)):
                        if done_indices[k] >= j:
                            done_batch[i, k] = torch.zeros_like(self.done_mem[0])
                    break
        return obs_batch, action_batch, reward_batch, done_batch
