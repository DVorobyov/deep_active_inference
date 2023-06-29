import torch
import torch.nn.functional as F
import numpy as np
import datetime
import gym
from Model import Model
from Memory import ReplayBuffer


class Agent():
    def __init__(self, run_id, device, env, n_episodes, max_iters, n_hidden_trans,
                 lr_trans, n_hidden_pol, lr_pol, n_hidden_val, lr_val, gamma, beta):
        self.max_iters = max_iters
        self.run_id = run_id
        self.device = device
        self.env = gym.make(env, render_mode="human")
        self.n_episodes = n_episodes
        self.n_hidden_trans = n_hidden_trans
        self.lr_trans = float(lr_trans)
        self.n_hidden_pol = n_hidden_pol
        self.lr_pol = float(lr_pol)
        self.n_hidden_val = n_hidden_val
        self.lr_val = float(lr_val)
        self.memory_capacity = 65536
        self.batch_size = 64
        self.freeze_period = 25
        self.gamma = float(gamma)
        self.Beta = float(beta)
        self.obs_shape = self.env.observation_space.shape
        self.obs_size = np.prod(self.obs_shape)
        self.n_actions = self.env.action_space.n
        self.freeze_cntr = 0
        self.transition_net = Model(self.obs_size + 1, self.obs_size, self.n_hidden_trans, lr=self.lr_trans,
                                    device=self.device)
        self.policy_net = Model(self.obs_size, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True,
                                device=self.device)
        self.value_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.memory = ReplayBuffer(self.memory_capacity, self.obs_shape, device=self.device)
        self.log_path = f"./logs/log{run_id}.txt"
        self.results_path = f"./results/result_{env}_{run_id}.npz"
        self.network_save_path = f"./networks/net_{env}_{run_id}.pth"
        self.load_network = False
        self.network_load_path = ""
        self.record = open(self.log_path, "a")

        if self.load_network:
            self.transition_net.load_state_dict(torch.load(self.network_load_path.format("trans")))
            self.transition_net.eval()
            self.policy_net.load_state_dict(torch.load(self.network_load_path.format("pol")))
            self.policy_net.eval()
            self.value_net.load_state_dict(torch.load(self.network_load_path.format("val")))
            self.value_net.eval()
        self.target_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.obs_indices = [2, 1, 0]
        self.action_indices = [2, 1]
        self.reward_indices = [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1

    def select_action(self, obs):
        with torch.no_grad():
            policy = self.policy_net(obs)
            return torch.multinomial(policy, 1)

    def get_mini_batches(self):
        all_obs_batch, all_actions_batch, reward_batch_t1, done_batch_t2 = self.memory.sample(
            self.obs_indices, self.action_indices, self.reward_indices,
            self.done_indices, self.max_n_indices, self.batch_size)
        obs_batch_t0 = all_obs_batch[:, 0].view([self.batch_size] + [dim for dim in self.obs_shape])
        obs_batch_t1 = all_obs_batch[:, 1].view([self.batch_size] + [dim for dim in self.obs_shape])
        obs_batch_t2 = all_obs_batch[:, 2].view([self.batch_size] + [dim for dim in self.obs_shape])
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)
        X = torch.cat((obs_batch_t0, action_batch_t0.float()), dim=1)
        pred_batch_t0t1 = self.transition_net(X)
        pred_error_batch_t0t1 = torch.mean(F.mse_loss(
            pred_batch_t0t1, obs_batch_t1, reduction='none'), dim=1).unsqueeze(1)
        return (obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
                action_batch_t1, reward_batch_t1, done_batch_t2, pred_error_batch_t0t1)

    def compute_value_net_loss(self, obs_batch_t1, obs_batch_t2,
                               action_batch_t1, reward_batch_t1,
                               done_batch_t2, pred_error_batch_t0t1):
        with torch.no_grad():
            policy_batch_t2 = self.policy_net(obs_batch_t2)
            target_EFEs_batch_t2 = self.target_net(obs_batch_t2)
            weighted_targets = ((1 - done_batch_t2) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)
            EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t0t1 + self.Beta * weighted_targets
        EFE_batch_t1 = self.value_net(obs_batch_t1).gather(1, action_batch_t1)
        value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)

        return value_net_loss

    def compute_VFE(self, obs_batch_t1, pred_error_batch_t0t1):
        policy_batch_t1 = self.policy_net(obs_batch_t1)
        EFEs_batch_t1 = self.value_net(obs_batch_t1).detach()
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1 - 1e-9)
        energy_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).view(self.batch_size, 1)
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).view(self.batch_size, 1)
        VFE_batch = pred_error_batch_t0t1 + (energy_batch - entropy_batch)
        VFE = torch.mean(VFE_batch)
        return VFE

    def learn(self):
        if self.memory.push_count - self.max_n_indices * 2 < self.batch_size:
            return
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
        self.freeze_cntr += 1
        (obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
         action_batch_t1, reward_batch_t1, done_batch_t2,
         pred_error_batch_t0t1) = self.get_mini_batches()
        value_net_loss = self.compute_value_net_loss(obs_batch_t1, obs_batch_t2,
                                                     action_batch_t1, reward_batch_t1,
                                                     done_batch_t2, pred_error_batch_t0t1)
        VFE = self.compute_VFE(obs_batch_t1, pred_error_batch_t0t1)
        self.transition_net.optimizer.zero_grad()
        self.policy_net.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()
        VFE.backward()
        value_net_loss.backward()
        self.transition_net.optimizer.step()
        self.policy_net.optimizer.step()
        self.value_net.optimizer.step()

    def train(self):
        msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
        print(msg)
        self.record.write(msg + "\n")
        results = []
        for ith_episode in range(self.n_episodes):
            total_reward = 0
            obs = self.env.reset()[0]
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            done = False
            reward = 0
            it = 0
            while not done:
                action = self.select_action(obs)
                try:
                    self.memory.push(obs, action, reward, done)
                except:
                    pass
                obs, reward, done, _ = self.env.step(action[0].item())[:4]
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                total_reward += reward
                self.learn()
                if done or it >= self.max_iters:
                    done = True
                    if it < self.max_iters:
                        self.memory.push(obs, -99, -99, done)
                    else:
                        self.memory.push(obs, action, reward, done)
                it += 1
            results.append(total_reward)
            if ith_episode > 0:
                msg = "Episodes: {:4d}, score: {:3.2f}".format(ith_episode, results[-1])
                print(msg)
                self.record.write(msg + "\n")
                if ith_episode % 1 == 0:
                    self.record.close()
                    self.record = open(self.log_path, "a")
            if ith_episode > 0 and ith_episode % 100 == 0:
                np.savez("results/intermediary/intermediary_results{}_{:d}".format(self.run_id, ith_episode),
                         np.array(results))
            if ith_episode > 0 and ith_episode % 500 == 0:
                torch.save(self.transition_net.state_dict(),
                           "networks/intermediary/intermediary_transnet{}_{:d}.pth".format(self.run_id, ith_episode))
                torch.save(self.policy_net.state_dict(),
                           "networks/intermediary/intermediary_polnet{}_{:d}.pth".format(self.run_id, ith_episode))
                torch.save(self.value_net.state_dict(),
                           "networks/intermediary/intermediary_valnet{}_{:d}.pth".format(self.run_id, ith_episode))
        self.env.close()
        np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
        np.savez(self.results_path, np.array(results))
        torch.save(self.transition_net.state_dict(),
                   "networks/intermediary/intermediary_transnet{}_end.pth".format(self.run_id))
        torch.save(self.policy_net.state_dict(),
                   "networks/intermediary/intermediary_polnet{}_end.pth".format(self.run_id))
        torch.save(self.value_net.state_dict(),
                   "networks/intermediary/intermediary_valnet{}_end.pth".format(self.run_id))
        torch.save(self.transition_net.state_dict(), self.network_save_path.format("trans"))
        torch.save(self.policy_net.state_dict(), self.network_save_path.format("pol"))
        torch.save(self.value_net.state_dict(), self.network_save_path.format("val"))
        msg = "Training finished at {}".format(datetime.datetime.now())
        print(msg)
        self.record.write(msg)
        self.record.close()
