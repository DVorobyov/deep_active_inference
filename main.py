from Agent import Agent


run_id = 15
device = 'cpu'
env = 'LunarLander-v2'
n_episodes = 4000
n_hidden_trans = 128
lr_trans = 1e-3
n_hidden_pol = 128
lr_pol = 1e-3
n_hidden_val = 64
lr_val = 1e-4
gamma = 1
beta = 0.99
max_iters = 1000

agent = Agent(run_id, device, env, n_episodes, max_iters, n_hidden_trans, lr_trans, n_hidden_pol, lr_pol, n_hidden_val,
              lr_val, gamma, beta)
agent.train()
