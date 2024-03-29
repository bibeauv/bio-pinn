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

    def __init__(self, device, E, A, e, c1, c2, neurons):

        super().__init__()

        self.activation = nn.Tanh()

        self.f1 = nn.Linear(1, neurons)
        self.f2 = nn.Linear(neurons, neurons)
        self.f3 = nn.Linear(neurons, neurons)
        self.out = nn.Linear(neurons, 6)

        self.E1 = torch.tensor(E[0], requires_grad=True).float().to(device)
        self.E2 = torch.tensor(E[1], requires_grad=True).float().to(device)
        self.E3 = torch.tensor(E[2], requires_grad=True).float().to(device)
        self.E4 = torch.tensor(E[3], requires_grad=True).float().to(device)
        self.E5 = torch.tensor(E[4], requires_grad=True).float().to(device)
        self.E6 = torch.tensor(E[5], requires_grad=True).float().to(device)
        
        self.A1 = torch.tensor(A[0], requires_grad=True).float().to(device)
        self.A2 = torch.tensor(A[1], requires_grad=True).float().to(device)
        self.A3 = torch.tensor(A[2], requires_grad=True).float().to(device)
        self.A4 = torch.tensor(A[3], requires_grad=True).float().to(device)
        self.A5 = torch.tensor(A[4], requires_grad=True).float().to(device)
        self.A6 = torch.tensor(A[5], requires_grad=True).float().to(device)

        self.E1 = nn.Parameter(self.E1)
        self.E2 = nn.Parameter(self.E2)
        self.E3 = nn.Parameter(self.E3)
        self.E4 = nn.Parameter(self.E4)
        self.E5 = nn.Parameter(self.E5)
        self.E6 = nn.Parameter(self.E6)
        
        self.A1 = nn.Parameter(self.A1)
        self.A2 = nn.Parameter(self.A2)
        self.A3 = nn.Parameter(self.A3)
        self.A4 = nn.Parameter(self.A4)
        self.A5 = nn.Parameter(self.A5)
        self.A6 = nn.Parameter(self.A6)

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

    def __init__(self, X, Y, idx, idx_y0, f_hat, learning_rate, E, A, e, c1, c2, neurons, regularization, device, prm):
        
        def loss_function_ode(output, target):
            
            loss = torch.mean((output - target)**2)

            return loss
        
        def loss_function_data(output, target):

            loss = torch.mean((output[idx] - target)**2)

            return loss
        
        def loss_function_IC(output, target):

            loss = torch.mean((output[idx_y0] - target[0])**2)

            return loss
        
        self.PINN = PINeuralNet(device, E, A, e, c1, c2, neurons).to(device)

        self.PINN.register_parameter('E1', self.PINN.E1)
        self.PINN.register_parameter('E2', self.PINN.E2)
        self.PINN.register_parameter('E3', self.PINN.E3)
        self.PINN.register_parameter('E4', self.PINN.E4)
        self.PINN.register_parameter('E5', self.PINN.E5)
        self.PINN.register_parameter('E6', self.PINN.E6)
        
        self.PINN.register_parameter('A1', self.PINN.A1)
        self.PINN.register_parameter('A2', self.PINN.A2)
        self.PINN.register_parameter('A3', self.PINN.A3)
        self.PINN.register_parameter('A4', self.PINN.A4)
        self.PINN.register_parameter('A5', self.PINN.A5)
        self.PINN.register_parameter('A6', self.PINN.A6)

        self.PINN.register_parameter('e', self.PINN.e)
        self.PINN.register_parameter('c1', self.PINN.c1)
        self.PINN.register_parameter('c2', self.PINN.c2)

        self.x = X
        self.y = Y

        self.loss_function_ode = loss_function_ode
        self.loss_function_data = loss_function_data
        self.loss_function_IC = loss_function_IC
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

        y = self.PINN(g)
        cTG = y[:,0].reshape(-1,1)
        cDG = y[:,1].reshape(-1,1)
        cMG = y[:,2].reshape(-1,1)
        cG = y[:,3].reshape(-1,1)
        cME = y[:,4].reshape(-1,1)
        T = y[:,5].reshape(-1,1)
        
        k1 = self.PINN.A1 + self.PINN.E1 * (T - 34)
        k2 = self.PINN.A2 + self.PINN.E2 * (T - 34)
        k3 = self.PINN.A3 + self.PINN.E3 * (T - 34)
        k4 = self.PINN.A4 + self.PINN.E4 * (T - 34)
        k5 = self.PINN.A5 + self.PINN.E5 * (T - 34)
        k6 = self.PINN.A6 + self.PINN.E6 * (T - 34)
        
        grad_cTG = autograd.grad(cTG, g, torch.ones(x.shape[0], 1).to(self.device), \
                                 retain_graph=True, create_graph=True) \
                                 [0]
        grad_cDG = autograd.grad(cDG, g, torch.ones(x.shape[0], 1).to(self.device), \
                                 retain_graph=True, create_graph=True) \
                                 [0]
        grad_cMG = autograd.grad(cMG, g, torch.ones(x.shape[0], 1).to(self.device), \
                                 retain_graph=True, create_graph=True) \
                                 [0]
        grad_cG = autograd.grad(cG, g, torch.ones(x.shape[0], 1).to(self.device), \
                                retain_graph=True, create_graph=True) \
                                [0]
        grad_cME = autograd.grad(cME, g, torch.ones(x.shape[0], 1).to(self.device), \
                                 retain_graph=True, create_graph=True) \
                                 [0]
        grad_T = autograd.grad(T, g, torch.ones(x.shape[0], 1).to(self.device), \
                               retain_graph=True, create_graph=True) \
                               [0]

        self.loss_cTG_ode = self.loss_function_ode(grad_cTG + k1*cTG - k2*cDG*cME, self.f_hat)
        self.loss_cDG_ode = self.loss_function_ode(grad_cDG - k1*cTG + k2*cDG*cME \
                                                            + k3*cDG - k4*cMG*cME, self.f_hat)
        self.loss_cMG_ode = self.loss_function_ode(grad_cMG - k3*cDG + k4*cMG*cME \
                                                            + k5*cMG - k6*cG*cME, self.f_hat)
        self.loss_cG_ode = self.loss_function_ode(grad_cG - k5*cMG + k6*cG*cME, self.f_hat)
        self.loss_cME_ode = self.loss_function_ode(grad_cME - k1*cTG + k2*cDG*cME \
                                                            - k3*cDG + k4*cMG*cME \
                                                            - k5*cMG + k6*cG*cME, self.f_hat)
        self.loss_T_ode = self.loss_function_ode(self.prm.m_Cp*grad_T - self.PINN.e*self.prm.Q - self.PINN.c1*T - self.PINN.c2, self.f_hat)
        
        self.loss_cTG_data = self.loss_function_data(cTG, y_train[:,0].reshape(-1,1))
        self.loss_cDG_data = self.loss_function_data(cDG, y_train[:,1].reshape(-1,1))
        self.loss_cMG_data = self.loss_function_data(cMG, y_train[:,2].reshape(-1,1))
        self.loss_cG_data = self.loss_function_IC(cG, y_train[:,3].reshape(-1,1))
        self.loss_cME_data = self.loss_function_IC(cME, y_train[:,4].reshape(-1,1))
        self.loss_T_data = self.loss_function_data(T, y_train[:,5].reshape(-1,1))

        self.loss_c_data = self.loss_cTG_data + self.loss_cDG_data + self.loss_cMG_data + self.loss_cG_data + self.loss_cME_data
        self.loss_c_ode = self.loss_cTG_ode + self.loss_cDG_ode + self.loss_cMG_ode + self.loss_cG_ode + self.loss_cME_ode

        self.loss_ode = self.loss_cTG_ode + self.loss_cDG_ode + self.loss_cMG_ode + self.loss_cG_ode + self.loss_cME_ode + self.loss_T_ode
        self.loss_data = self.loss_cTG_data + self.loss_cDG_data + self.loss_cMG_data + self.loss_cG_data + self.loss_cME_data + self.loss_T_data
        
        self.total_loss = self.regularization * self.loss_ode + self.loss_data
        
        return self.total_loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        loss = self.loss(self.x, self.y)
        
        loss.backward()
        
        return loss