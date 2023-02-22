# Importation de librairies
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
ALPHA = 0.5
PATH = os.getcwd() + '/model.pt'

# PINN architecture
class PINeuralNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.activation = nn.Tanh()

        self.f1 = nn.Linear(2, NEURONS)
        self.f2 = nn.Linear(NEURONS, NEURONS)
        self.f3 = nn.Linear(NEURONS, NEURONS)
        self.out = nn.Linear(NEURONS, 3)

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

    def __init__(self, X, Y, y0, idx, idx_y0, f_hat, device, prm):
        
        def loss_function_ode(output, target):
            
            loss = torch.mean((output - target)**2)

            return loss

        def loss_function_c_data(output, target):

            loss = torch.mean((output[idx_y0] - target)**2)

            return loss
        
        def loss_function_T_data(output, target):

            loss = torch.mean((output[idx] - target[idx])**2)

            return loss
        
        self.PINN = PINeuralNet().to(device)

        self.x = X
        self.y = Y
        self.y0 = y0

        self.loss_function_ode = loss_function_ode
        self.loss_function_c_data = loss_function_c_data
        self.loss_function_T_data = loss_function_T_data
        self.f_hat = f_hat

        self.device = device
        self.lr = LEARNING_RATE
        self.params = list(self.PINN.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        self.prm = prm

    def loss(self, x, y_train):

        g = x.clone()
        g.requires_grad = True

        t = g[:,0].reshape(-1,1)
        Q = g[:,1].reshape(-1,1)

        y = self.PINN(g)

        cTG = y[:,0].reshape(-1,1)
        T = y[:,1].reshape(-1,1)
        k = y[:,2].reshape(-1,1)
        
        grad_cTG = autograd.grad(cTG, g, torch.ones(x.shape[0], 1).to(self.device), \
                                 retain_graph=True, create_graph=True) \
                                 [0][:,0].reshape(-1,1)
        
        grad_T = autograd.grad(T, g, torch.ones(x.shape[0], 1).to(self.device), \
                               retain_graph=True, create_graph=True) \
                               [0][:,0].reshape(-1,1)

        self.loss_cTG_ode = self.loss_function_ode(grad_cTG + k*cTG, self.f_hat)
        self.loss_T_ode = self.loss_function_ode(grad_T * self.prm.m * self.prm.Cp - \
                                                      self.prm.epsilon*Q - \
                                                      self.prm.dHrx * self.prm.V/1000 * grad_cTG, self.f_hat)
        
        self.loss_cTG_data = self.loss_function_c_data(cTG, self.y0)
        self.loss_T_data = self.loss_function_T_data(T, y_train)
        
        self.total_loss = self.loss_cTG_ode + self.loss_T_ode + self.loss_T_data + 1e6*self.loss_cTG_data
        
        return self.total_loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        loss = self.loss(self.x, self.y)
        
        loss.backward()
        
        return loss
    
    def save_model(self, PATH):
         
         torch.save(self.PINN, PATH)
    
# Full PINN to discover Ea & A
class Discovery():

    def __init__(self, X, Y, y0, idx, idx_y0, f_hat, device, prm, Ea, A):
        
        def loss_function_ode(output, target):
            
            loss = torch.mean((output - target)**2)

            return loss

        def loss_function_c_data(output, target):

            loss = torch.mean((output[idx_y0] - target)**2)

            return loss
        
        def loss_function_T_data(output, target):

            loss = torch.mean((output[idx] - target[idx])**2)

            return loss
        
        self.PINN = torch.load(PATH)
        self.PINN.eval()

        self.Ea = torch.tensor(Ea, requires_grad=True).float().to(device)
        self.A = torch.tensor(A, requires_grad=True).float().to(device)
        self.Ea = nn.Parameter(self.Ea)
        self.A = nn.Parameter(self.A)
        self.PINN.register_parameter('Ea', self.Ea)
        self.PINN.register_parameter('A', self.A)

        self.x = X
        self.y = Y
        self.y0 = y0

        self.loss_function_ode = loss_function_ode
        self.loss_function_c_data = loss_function_c_data
        self.loss_function_T_data = loss_function_T_data
        self.f_hat = f_hat

        self.device = device
        self.lr = LEARNING_RATE
        self.params = list(self.PINN.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

        self.prm = prm

    def loss(self, x, y_train):

        g = x.clone()
        g.requires_grad = True

        t = g[:,0].reshape(-1,1)
        Q = g[:,1].reshape(-1,1)

        y = self.PINN(g)

        cTG = y[:,0].reshape(-1,1)
        T = y[:,1].reshape(-1,1)
        k = y[:,2].reshape(-1,1)
        
        grad_cTG = autograd.grad(cTG, g, torch.ones(x.shape[0], 1).to(self.device), \
                                 retain_graph=True, create_graph=True) \
                                 [0][:,0].reshape(-1,1)
        
        grad_T = autograd.grad(T, g, torch.ones(x.shape[0], 1).to(self.device), \
                               retain_graph=True, create_graph=True) \
                               [0][:,0].reshape(-1,1)

        self.loss_cTG_ode = self.loss_function_ode(grad_cTG + k*cTG, self.f_hat)
        self.loss_T_ode = self.loss_function_ode(grad_T * self.prm.m * self.prm.Cp - \
                                                      self.prm.epsilon*Q - \
                                                      self.prm.dHrx * self.prm.V/1000 * grad_cTG, self.f_hat)
        
        self.loss_cTG_data = self.loss_function_c_data(cTG, self.y0)
        self.loss_T_data = self.loss_function_T_data(T, y_train)

        self.loss_k = self.loss_function_ode(k - self.A*torch.exp(-self.Ea/T), self.f_hat)
        
        self.total_loss = self.loss_cTG_ode + self.loss_T_ode + self.loss_T_data + 1e6*self.loss_cTG_data + self.loss_k
        
        return self.total_loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        loss = self.loss(self.x, self.y)
        
        loss.backward()
        
        return loss