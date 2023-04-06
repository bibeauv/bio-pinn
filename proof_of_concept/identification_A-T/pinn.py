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

# Params
NEURONS = 20
LEARNING_RATE = 1e-3
ALPHA = 0.5
PATH = os.getcwd() + '/model.pt'

# PINN architecture
class PINeuralNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.activation = nn.Tanh()
        # self.activation_out = nn.ReLU()

        self.f1 = nn.Linear(1, NEURONS)
        self.f2 = nn.Linear(NEURONS, NEURONS)
        self.f3 = nn.Linear(NEURONS, NEURONS)
        self.out = nn.Linear(NEURONS, 2)

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
        # a_4 = self.activation_out(a_4)
        
        return a_4

# Full PINN to discover k
class Curiosity():

    def __init__(self, X, Y, idx, f_hat, device, prm):
        
        def loss_function_ode(output, target):
            
            loss = torch.mean((output - target)**2)

            return loss
        
        def loss_function_data(output, target):

            loss = torch.mean((output[idx] - target[idx])**2)

            return loss
        
        self.PINN = PINeuralNet().to(device)

        self.Ea = torch.tensor(1.0, requires_grad=True).float().to(device)
        self.A = torch.tensor(1.0, requires_grad=True).float().to(device)
        self.Ea = nn.Parameter(self.Ea)
        self.A = nn.Parameter(self.A)
        self.PINN.register_parameter('Ea', self.Ea)
        self.PINN.register_parameter('A', self.A)

        self.e = torch.tensor(1.0, requires_grad=True).float().to(device)
        self.c1 = torch.tensor(1.0, requires_grad=True).float().to(device)
        self.c2 = torch.tensor(1.0, requires_grad=True).float().to(device)
        self.e = nn.Parameter(self.e)
        self.c1 = nn.Parameter(self.c1)
        self.c2 = nn.Parameter(self.c2)
        self.PINN.register_parameter('e', self.e)
        self.PINN.register_parameter('c1', self.c1)
        self.PINN.register_parameter('c2', self.c2)

        # self.k = torch.ones(len(X), requires_grad=True).float().to(device)
        # self.k = nn.Parameter(self.k)
        # self.PINN.register_parameter('k', self.k)

        self.x = X
        self.y = Y

        self.loss_function_ode = loss_function_ode
        self.loss_function_data = loss_function_data
        self.f_hat = f_hat

        self.device = device
        self.lr = LEARNING_RATE
        self.params = list(self.PINN.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        self.prm = prm

    def loss(self, x, y_train):

        g = x.clone()
        g.requires_grad = True

        y = self.PINN(g)
        c = y[:,0].reshape(-1,1)
        T = y[:,1].reshape(-1,1)
        k = self.A * torch.exp(-self.Ea / T)
        
        grad_c = autograd.grad(c, g, torch.ones(x.shape[0], 1).to(self.device), \
                               retain_graph=True, create_graph=True) \
                               [0]
        
        grad_T = autograd.grad(T, g, torch.ones(x.shape[0], 1).to(self.device), \
                               retain_graph=True, create_graph=True) \
                               [0]

        self.loss_c_ode = self.loss_function_ode(grad_c + k*c, self.f_hat)
        self.loss_T_ode = self.loss_function_ode(self.prm.m_Cp*grad_T - self.e*self.prm.Q - self.prm.dHrx * grad_c * self.prm.V - self.c1*T - self.c2, self.f_hat)
        
        self.loss_c_data = self.loss_function_data(c, y_train[:,0].reshape(-1,1))
        self.loss_T_data = self.loss_function_data(T, y_train[:,1].reshape(-1,1))

        # self.loss_k = self.loss_function_ode(k - self.A * torch.exp(-self.Ea / T), self.f_hat)
        
        self.total_loss = self.loss_c_ode + self.loss_c_data + self.loss_T_ode + self.loss_T_data
        
        return self.total_loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        loss = self.loss(self.x, self.y)
        
        loss.backward()
        
        return loss
    
    def save_model(self, PATH):
         
         torch.save(self.PINN, PATH)