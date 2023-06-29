import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=1e-3, softmax=False, device='cpu'):
        super(Model, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.softmax = softmax
        self.fc1 = nn.Linear(self.n_inputs, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.fc3 = nn.Linear(self.n_hidden, self.n_outputs)
        self.optimizer = optim.Adam(self.parameters(), lr)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        h_relu = F.gelu(self.fc1(x))
        h_relu = F.gelu(self.fc2(h_relu))
        y = self.fc3(h_relu)
        if self.softmax:
            y = F.softmax(self.fc3(h_relu), dim=-1).clamp(min=1e-9, max=1 - 1e-9)
        return y