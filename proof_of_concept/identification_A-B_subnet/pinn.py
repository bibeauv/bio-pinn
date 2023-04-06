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
NEURONS = 10
LEARNING_RATE = 1e-3
PATH = os.getcwd() + '/model.pt'

# PINN architecture
class PINeuralNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.activation = nn.Tanh()

        self.f1 = nn.Linear(1, NEURONS)
        self.f2 = nn.Linear(NEURONS, NEURONS)
        self.f3 = nn.Linear(NEURONS, NEURONS)
        self.out = nn.Linear(NEURONS, 1)

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

# Full PINN to predict cA
class subnetA():

    def __init__(self, X, Y, idx, f_hat, device, prm, kinetics):
        
        def loss_function_ode(output, target):
            
            loss = torch.mean((output - target)**2)

            return loss
        
        def loss_function_data(output, target):

            loss = torch.mean((output[idx] - target[idx])**2)

            return loss
        
        self.PINN = PINeuralNet().to(device)

        self.E1 = torch.tensor(kinetics[0], requires_grad=True).float().to(device)
        self.A1 = torch.tensor(kinetics[1], requires_grad=True).float().to(device)
        self.E2 = torch.tensor(kinetics[2], requires_grad=True).float().to(device)
        self.A2 = torch.tensor(kinetics[3], requires_grad=True).float().to(device)

        self.E1 = nn.Parameter(self.E1)
        self.A1 = nn.Parameter(self.A1)
        self.E2 = nn.Parameter(self.E2)
        self.A2 = nn.Parameter(self.A2)

        self.PINN.register_parameter('E1', self.E1)
        self.PINN.register_parameter('A1', self.A1)
        self.PINN.register_parameter('E2', self.E2)
        self.PINN.register_parameter('A2', self.A2)

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

    def add_other_PINN(self, PINNB):

        self.PINNB = PINNB

    def loss(self, x, y_train):

        g = x.clone()
        g.requires_grad = True

        self.c = self.PINN(g)
        T = y_train[:,2].reshape(-1,1)
        k1 = self.A1 * torch.exp(-self.E1 / T)
        k2 = self.A2 * torch.exp(-self.E2 / T)
        
        grad_c = autograd.grad(self.c, g, torch.ones(x.shape[0], 1).to(self.device), \
                               retain_graph=True, create_graph=True) \
                               [0]

        self.loss_c_ode = self.loss_function_ode(grad_c + k1*self.c - k2*self.PINNB.c.detach(), self.f_hat)
        
        self.loss_c_data = self.loss_function_data(self.c, y_train[:,0].reshape(-1,1))
        
        self.total_loss = self.loss_c_ode + self.loss_c_data
        
        return self.total_loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        loss = self.loss(self.x, self.y)
        
        loss.backward()
        
        return loss
    
    def save_model(self, PATH):
         
         torch.save(self.PINN, PATH)

# Full PINN to predict cB
class subnetB():

    def __init__(self, X, Y, idx, f_hat, device, prm, kinetics):
        
        def loss_function_ode(output, target):
            
            loss = torch.mean((output - target)**2)

            return loss
        
        def loss_function_data(output, target):

            loss = torch.mean((output[idx] - target[idx])**2)

            return loss
        
        self.PINN = PINeuralNet().to(device)

        self.E1 = torch.tensor(kinetics[0], requires_grad=True).float().to(device)
        self.A1 = torch.tensor(kinetics[1], requires_grad=True).float().to(device)
        self.E2 = torch.tensor(kinetics[2], requires_grad=True).float().to(device)
        self.A2 = torch.tensor(kinetics[3], requires_grad=True).float().to(device)

        self.E1 = nn.Parameter(self.E1)
        self.A1 = nn.Parameter(self.A1)
        self.E2 = nn.Parameter(self.E2)
        self.A2 = nn.Parameter(self.A2)

        self.PINN.register_parameter('E1', self.E1)
        self.PINN.register_parameter('A1', self.A1)
        self.PINN.register_parameter('E2', self.E2)
        self.PINN.register_parameter('A2', self.A2)

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

    def add_other_PINN(self, PINNA):

        self.PINNA = PINNA

    def loss(self, x, y_train):

        g = x.clone()
        g.requires_grad = True

        self.c = self.PINN(g)
        T = y_train[:,2].reshape(-1,1)
        k1 = self.A1 * torch.exp(-self.E1 / T)
        k2 = self.A2 * torch.exp(-self.E2 / T)
        
        grad_c = autograd.grad(self.c, g, torch.ones(x.shape[0], 1).to(self.device), \
                               retain_graph=True, create_graph=True) \
                               [0]

        self.loss_c_ode = self.loss_function_ode(grad_c + k2*self.c - k1*self.PINNA.c.detach(), self.f_hat)
        
        self.loss_c_data = self.loss_function_data(self.c, y_train[:,1].reshape(-1,1))
        
        self.total_loss = self.loss_c_ode + self.loss_c_data
        
        return self.total_loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        loss = self.loss(self.x, self.y)
        
        loss.backward()
        
        return loss
    
    def save_model(self, PATH):
         
         torch.save(self.PINN, PATH)