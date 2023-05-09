# Libraries importation
import torch
import torch.autograd as autograd
import torch.nn as nn

import numpy as np
import os

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# PINN architecture
class PINeuralNet(nn.Module):

    def __init__(self, device, e, c1, c2, neurons):

        super().__init__()

        self.activation = nn.Tanh()

        self.f1 = nn.Linear(2, neurons)
        self.f2 = nn.Linear(neurons, neurons)
        self.f3 = nn.Linear(neurons, neurons)
        self.out = nn.Linear(neurons, 1)

        self.e = torch.tensor(e, requires_grad=True).float().to(device)
        self.c1 = torch.tensor(c1, requires_grad=True).float().to(device)
        self.c2 = torch.tensor(c2, requires_grad=True).float().to(device)

        self.e = nn.Parameter(self.e)
        self.c1 = nn.Parameter(self.c1)
        self.c2 = nn.Parameter(self.c2)

    def forward(self, x):

        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        
        a = x.float()
        
        z_1 = self.f1(a)
        a_1 = self.activation(z_1)
        z_2 = self.f2(a_1)
        a_2 = self.activation(z_2)
        z_3 = self.f3(a_2)
        a_3 = self.activation(z_3)

        a_4 = self.out(a_3)
        
        return a_4

# Full PINN to discover k
class Curiosity():

    def __init__(self, X, Y, f_hat, learning_rate, e, c1, c2, neurons, regularization, device, prm):
        
        def loss_function_ode(output, target):
            
            loss = torch.mean((output - target)**2)

            return loss
        
        def loss_function_data(output, target):

            loss = torch.mean((output - target)**2)

            return loss
        
        self.PINN = PINeuralNet(device, e, c1, c2, neurons).to(device)

        self.PINN.register_parameter('e', self.PINN.e)
        self.PINN.register_parameter('c1', self.PINN.c1)
        self.PINN.register_parameter('c2', self.PINN.c2)

        self.x = X
        self.y = Y

        self.loss_function_ode = loss_function_ode
        self.loss_function_data = loss_function_data
        self.f_hat = f_hat
        self.regularization = regularization

        self.device = device
        self.lr = learning_rate
        self.params = list(self.PINN.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        self.prm = prm

    def loss(self, x, y_train):

        g = x.clone()
        g.requires_grad = True

        t = g[:,0].reshape(-1,1)
        Q = g[:,1].reshape(-1,1)

        T = self.PINN(g)

        grad_T = autograd.grad(T, g, torch.ones(x.shape[0], 1).to(self.device), \
                               retain_graph=True, create_graph=True) \
                               [0][:,0].reshape(-1,1)

        self.loss_T_ode = self.loss_function_ode(self.prm.m_Cp*grad_T - self.PINN.e*Q - self.PINN.c1*T - self.PINN.c2, self.f_hat)
        
        self.loss_T_data = self.loss_function_data(T, y_train[:,0].reshape(-1,1))
        
        self.total_loss = self.regularization * self.loss_T_ode + self.loss_T_data
        
        return self.total_loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        loss = self.loss(self.x, self.y)
        
        loss.backward()
        
        return loss