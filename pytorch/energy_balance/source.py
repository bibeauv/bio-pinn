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
LEARNING_RATE = 1e-2
NEURONS = 10
ALPHA = 0.5

class subDNN(nn.Module):
    
    def __init__(self):

        super().__init__()
        
        self.activation = nn.Tanh()
        
        self.f1 = nn.Linear(1, NEURONS)
        self.f2 = nn.Linear(NEURONS, NEURONS)
        self.f3 = nn.Linear(NEURONS, NEURONS)
        self.out = nn.Linear(NEURONS, 1)

    def initialize_parameters(self, device, k_pred):

        self.Ea = torch.tensor([k_pred[0]], requires_grad=True).float().to(device)
        self.A = torch.tensor([k_pred[1]], requires_grad=True).float().to(device)
        
        self.Ea = nn.Parameter(self.Ea)
        self.A = nn.Parameter(self.A)

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
    
class cTGNN():
    
    def __init__(self, device, f_hat, y_pred, k_pred, t_train, cTG_train, idx):

        self.dnn = subDNN().to(device)

        def loss_function_ode(output, target):
        
            loss = torch.mean((output - target)**2)

            return loss

        def loss_function_data(output, target):

            loss = torch.mean((output[idx] - target[idx])**2)

            return loss

        self.loss_function_ode = loss_function_ode
        self.loss_function_data = loss_function_data

        self.dnn.initialize_parameters(device, k_pred)
        
        self.dnn.register_parameter('Ea', self.dnn.Ea)
        self.dnn.register_parameter('A', self.dnn.A)
        
        self.device = device
        self.alpha = ALPHA
        self.f_hat = f_hat
        self.y_pred = y_pred
        self.k_pred = k_pred
        self.t_train = t_train
        self.cTG_train = cTG_train
        self.grad_cTG_pred = None
        self.idx = idx
        
        self.lr = LEARNING_RATE
        self.params = list(self.dnn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
    
    def pred(self, y_pred, k_pred):
        
        self.Temp = y_pred['Temp'].reshape(-1,1)

        with torch.no_grad():
            for i, p in enumerate(self.dnn.parameters()):
                if i == 0 or i == 1:
                    p.data.fill_(k_pred[i])
    
    def loss(self, x, y):

        g = x.clone()
        g.requires_grad = True

        self.cTG = self.dnn(g)
        
        k1 = self.dnn.A * torch.exp(-self.dnn.Ea / self.Temp)
        
        grad_cTG = autograd.grad(self.cTG, g, torch.ones(x.shape[0], 1).to(self.device), \
                                 retain_graph=True, create_graph=True)[0]
        self.y_pred['grad_cTG'] = grad_cTG.detach().clone()
        loss_cTG_ode = self.loss_function_ode(grad_cTG + k1*self.cTG, self.f_hat)
        
        loss_cTG_data = self.loss_function_data(self.cTG, y)
        
        loss = self.alpha*loss_cTG_ode + (1-self.alpha)*loss_cTG_data
        
        return loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        self.pred(self.y_pred, self.k_pred)
        
        x_train = self.t_train
        loss = self.loss(x_train, self.cTG_train)
        
        loss.backward()
        
        return loss

class TempNN():
    
    def __init__(self, device, f_hat, y_pred, k_pred, t_train, T_train, prm_mw, idx):

        self.dnn = subDNN().to(device)

        def loss_function_ode(output, target):
        
            loss = torch.mean((output - target)**2)

            return loss

        def loss_function_data(output, target):

            loss = torch.mean((output[idx] - target[idx])**2)

            return loss

        self.loss_function_ode = loss_function_ode
        self.loss_function_data = loss_function_data

        self.dnn.initialize_parameters(device, k_pred)
        
        self.dnn.register_parameter('Ea', self.dnn.Ea)
        self.dnn.register_parameter('A', self.dnn.A)
        
        self.device = device
        self.alpha = ALPHA
        self.f_hat = f_hat
        self.y_pred = y_pred
        self.k_pred = k_pred
        self.t_train = t_train
        self.T_train = T_train
        self.prm_mw = prm_mw
        self.idx = idx
        
        self.lr = LEARNING_RATE
        self.params = list(self.dnn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
    
    def pred(self, y_pred, k_pred):

        self.grad_cTG = y_pred['grad_cTG']
        
        with torch.no_grad():
            for i, p in enumerate(self.dnn.parameters()):
                if i == 0 or i == 1:
                    p.data.fill_(k_pred[i])
    
    def loss(self, x, y):

        g = x.clone()
        g.requires_grad = True

        self.Temp = self.dnn(g)
        
        grad_Temp = autograd.grad(self.Temp, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]

        loss_Temp_ode = self.loss_function_ode(grad_Temp - self.prm_mw.Q - self.prm_mw.dHrx * self.grad_cTG * self.prm_mw.V, self.f_hat)
        
        loss_Temp_data = self.loss_function_data(self.Temp, y)
        
        loss = self.alpha*loss_Temp_ode + (1-self.alpha)*loss_Temp_data
        
        return loss
    
    def closure(self):

        self.optimizer.zero_grad()
        
        self.pred(self.y_pred, self.k_pred)
        
        x_train = self.t_train
        loss = self.loss(x_train, self.T_train)
        
        loss.backward()
        
        return loss

def train_cNN(cNN, NN_index, y_pred, k_pred, t_train):
    
    cNN.optimizer.step(cNN.closure)
    for i, p in enumerate(cNN.dnn.parameters()):
        if i == 0 or i == 1:
            p.data.clamp_(min=0.)
            k_pred[i] = float(p.detach().cpu().numpy())
            
    x_train = t_train
    y_pred[NN_index] = cNN.dnn(x_train).detach().clone().flatten()
