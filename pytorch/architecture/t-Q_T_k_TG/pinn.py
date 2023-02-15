# Importation de librairies
import torch
import torch.autograd as autograd
import torch.nn as nn

import numpy as np

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# Params
NEURONS = 10
LEARNING_RATE = 1e-3
ALPHA = 0.5

# Activation
def inv_activation(input):
    
    return 1/input

def exp_activation(input):

    return torch.exp(input)

# PINN architecture
class PINeuralNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.activation = nn.Tanh()
        self.inv_activation = inv_activation
        self.exp_activation = exp_activation

        self.f1 = nn.Linear(2, NEURONS)
        self.f2 = nn.Linear(NEURONS, NEURONS)
        self.f3 = nn.Linear(NEURONS, NEURONS)
        self.out_T = nn.Linear(NEURONS, 1)
        self.inv_layer = nn.Linear(1, 1)
        self.exp_layer = nn.Linear(1, 1)
        self.out_k = nn.Linear(1, 1)
        self.out = nn.Linear(1, 1)

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

        a_4 = self.out_T(a_3)
        
        z_5 = self.inv_layer(a_4)
        a_5 = self.inv_activation(z_5)
        z_6 = self.exp_layer(a_5)
        a_6 = self.exp_activation(z_6)

        a_7 = self.out(a_6)

        self.a_ = [a_1, a_2, a_3, a_4, a_5, a_6, a_7]
        
        return a_7

# Full PINN
class TGNeuralNet():

    def __init__(self, X, Y, idx, f_hat, device):
        
        def loss_function_ode(output, target):
            
                loss = torch.mean((output - target)**2)

                return loss

        def loss_function_data(output, target):

            loss = torch.mean((output[idx] - target[idx])**2)

            return loss
        
        self.PINN = PINeuralNet().to(device)

        self.x = X
        self.y = Y

        self.loss_function_ode = loss_function_ode
        self.loss_function_data = loss_function_data
        self.f_hat = f_hat

        self.device = device
        self.lr = LEARNING_RATE
        self.params = list(self.PINN.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)

    def loss(self, x, y):

        g = x.clone()
        g.requires_grad = True

        self.cTG = self.PINN(g)
        
        grad_cTG = autograd.grad(self.cTG, g, torch.ones(x.shape[0], 1).to(self.device), \
                                 retain_graph=True, create_graph=True)[0][:,0].reshape(-1,1)
        
        k = self.PINN.a_[5]

        loss_cTG_ode = self.loss_function_ode(grad_cTG + k*self.cTG, self.f_hat)
        
        loss_cTG_data = self.loss_function_data(self.cTG, y[:,0].reshape(-1,1))
        
        loss = loss_cTG_ode + loss_cTG_data
        
        return loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        loss = self.loss(self.x, self.y)
        
        loss.backward()
        
        return loss