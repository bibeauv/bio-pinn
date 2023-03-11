# Importation de librairies
import torch
import torch.autograd as autograd
import torch.nn as nn

import numpy as np

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

class subDNN(nn.Module):
    
    def __init__(self, neurons, activation, params, idx, device):

        super().__init__()
        
        self.activation = activation
        self.activation_out = nn.ReLU()

        self.f1 = nn.Linear(2, neurons)
        self.f2 = nn.Linear(neurons, neurons)
        self.f3 = nn.Linear(neurons, neurons)
        self.out = nn.Linear(neurons, 1)
        
        self.k1 = nn.Parameter(params[0])
        # self.k2 = nn.Parameter(params[1])
        # self.k3 = nn.Parameter(params[2])
        # self.k4 = nn.Parameter(params[3])
        # self.k5 = nn.Parameter(params[4])
        # self.k6 = nn.Parameter(params[5])

        self.idx = idx

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
        a_4 = self.activation_out(a_4)
        
        return a_4
    
    def loss_function_ode(self, output, target):

        loss = torch.mean((output - target)**2)

        return loss
    
    def loss_function_data(self, output, target):

        loss = torch.mean((output[self.idx] - target[self.idx])**2)

        return loss
    
    def loss_function_IC(self, output, target):

        loss = torch.mean((output[0] - target[0])**2)

        return loss

class PINN():

    def __init__(self, neurons, activation, params, idx, X_train, Y_train, f_hat, leaning_rate, device):

        self.nn = subDNN(neurons, activation, params, idx, device).to(device)

        self.loss_function_ode = self.nn.loss_function_ode
        self.loss_function_data = self.nn.loss_function_data
        self.loss_function_IC = self.nn.loss_function_IC
        self.idx = idx

        self.nn.register_parameter('k1', self.nn.k1)
        # self.nn.register_parameter('k2', self.nn.k2)
        # self.nn.register_parameter('k3', self.nn.k3)
        # self.nn.register_parameter('k4', self.nn.k4)
        # self.nn.register_parameter('k5', self.nn.k5)
        # self.nn.register_parameter('k6', self.nn.k6)

        self.X_train = X_train
        self.Y_train = Y_train
        self.f_hat = f_hat
        self.device = device

        self.lr = leaning_rate
        self.params = list(self.nn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def loss(self, x, y):

        g = x.clone()
        g.requires_grad = True

        self.c = self.nn(g)
        # self.cB = self.c[:,0].reshape(-1,1)
        self.cTG = self.c[:,0].reshape(-1,1)
        # self.cDG = self.c[:,2].reshape(-1,1)
        # self.cMG = self.c[:,3].reshape(-1,1)
        # self.cG = self.c[:,4].reshape(-1,1)
        # self.T = self.c[:,1].reshape(-1,1)
        # self.k1 = self.c[:,1].reshape(-1,1)
        # self.k2 = self.c[:,6].reshape(-1,1)
        # self.k3 = self.c[:,7].reshape(-1,1)
        # self.k4 = self.c[:,8].reshape(-1,1)
        # self.k5 = self.c[:,9].reshape(-1,1)
        # self.k6 = self.c[:,10].reshape(-1,1)

        # # Loss biodiesel
        # grad_cB = autograd.grad(self.cB, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]

        # loss_cB_ode = self.loss_function_ode(grad_cB - self.nn.k1*self.cTG + self.nn.k2*self.cDG*self.cB \
        #                                              - self.nn.k3*self.cDG + self.nn.k4*self.cMG*self.cB \
        #                                              - self.nn.k5*self.cMG + self.nn.k6*self.cG*self.cB, self.f_hat)
        
        # loss_cB_IC = self.loss_function_IC(self.cB, y[:,0].reshape(-1,1))

        # Loss TG
        grad_cTG = autograd.grad(self.cTG, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0][:,0].reshape(-1,1)

        loss_cTG_ode = self.loss_function_ode(grad_cTG + self.nn.k1*self.cTG, self.f_hat)
        
        loss_cTG_data = self.loss_function_data(self.cTG, y[:,1].reshape(-1,1))

        loss_cTG_IC = self.loss_function_IC(self.cTG, y[:,1].reshape(-1,1))

        # # Loss DG
        # grad_cDG = autograd.grad(self.cDG, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]

        # loss_cDG_ode = self.loss_function_ode(grad_cDG - self.nn.k1*self.cTG + self.nn.k2*self.cDG*self.cB \
        #                                                + self.nn.k3*self.cDG - self.nn.k4*self.cMG*self.cB, self.f_hat)
        
        # loss_cDG_data = self.loss_function_data(self.cDG, y[:,2].reshape(-1,1))

        # loss_cDG_IC = self.loss_function_IC(self.cDG, y[:,2].reshape(-1,1))

        # # Loss MG
        # grad_cMG = autograd.grad(self.cMG, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]

        # loss_cMG_ode = self.loss_function_ode(grad_cMG - self.nn.k3*self.cDG + self.nn.k4*self.cMG*self.cB \
        #                                                + self.nn.k5*self.cMG - self.nn.k6*self.cG*self.cB, self.f_hat)
        
        # loss_cMG_data = self.loss_function_data(self.cMG, y[:,3].reshape(-1,1))

        # loss_cMG_IC = self.loss_function_IC(self.cMG, y[:,3].reshape(-1,1))

        # # Loss G
        # grad_cG = autograd.grad(self.cG, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]
        
        # loss_cG_ode = self.loss_function_ode(grad_cG - self.nn.k5*self.cMG + self.nn.k6*self.cG*self.cB, self.f_hat)
        
        # loss_cG_IC = self.loss_function_IC(self.cG, y[:,4].reshape(-1,1))

        # Loss Temperature
        t = g[:,0].detach().flatten()
        T = g[:,1].detach().flatten()
        Q = y[:,6].detach().flatten()

        grad_T = torch.gradient(T, spacing=(t,))[0].reshape(-1,1)
        # grad_T = autograd.grad(self.T, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]

        loss_T_ode = self.loss_function_ode(5.2*1.7*grad_T - 0.1*Q + 45650*6.3/1000*grad_cTG + 0.2, self.f_hat)

        # loss_T_data = self.loss_function_data(self.T, y[:,5].reshape(-1,1))

        # loss_T_IC = self.loss_function_IC(self.T, y[:,5].reshape(-1,1))
        
        # self.loss_ode = loss_cB_ode + loss_cTG_ode + loss_cDG_ode + loss_cMG_ode + loss_cG_ode + loss_T_ode
        # self.loss_data = loss_cTG_data + loss_cDG_data + loss_cMG_data
        # self.loss_IC = loss_cB_IC + loss_cTG_IC + loss_cDG_IC + loss_cMG_IC + loss_cG_IC

        self.loss_ode = loss_cTG_ode + 1e6*loss_T_ode
        self.loss_data = loss_cTG_data
        self.loss_IC = loss_cTG_IC

        self.total_loss = self.loss_ode + self.loss_data + self.loss_IC

        return self.total_loss
    
    def closure(self):

        self.optimizer.zero_grad()
                
        loss = self.loss(self.X_train, self.Y_train)
        
        loss.backward()
        
        return loss