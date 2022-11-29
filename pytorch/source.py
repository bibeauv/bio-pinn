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
ALPHA = 0.9

class subDNN(nn.Module):
    
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
    
class cTGNN():
    
    def __init__(self, device, f_hat, c_pred, k_pred, t_train, cTG_train, idx):
        self.dnn = subDNN().to(device)
        self.loss_function = nn.MSELoss(reduction = 'mean')

        self.k1 = torch.tensor([k_pred[0]], requires_grad=True).float().to(device)
        self.k2 = torch.tensor([k_pred[1]], requires_grad=True).float().to(device)
        self.k3 = torch.tensor([k_pred[2]], requires_grad=True).float().to(device)
        self.k4 = torch.tensor([k_pred[3]], requires_grad=True).float().to(device)
        self.k5 = torch.tensor([k_pred[4]], requires_grad=True).float().to(device)
        self.k6 = torch.tensor([k_pred[5]], requires_grad=True).float().to(device)
        
        self.k1 = nn.Parameter(self.k1)
        self.k2 = nn.Parameter(self.k2)
        self.k3 = nn.Parameter(self.k3)
        self.k4 = nn.Parameter(self.k4)
        self.k5 = nn.Parameter(self.k5)
        self.k6 = nn.Parameter(self.k6)
        
        self.dnn.register_parameter('k1', self.k1)
        self.dnn.register_parameter('k2', self.k2)
        self.dnn.register_parameter('k3', self.k3)
        self.dnn.register_parameter('k4', self.k4)
        self.dnn.register_parameter('k5', self.k5)
        self.dnn.register_parameter('k6', self.k6)
        
        self.device = device
        self.alpha = ALPHA
        self.f_hat = f_hat
        self.c_pred = c_pred
        self.k_pred = k_pred
        self.t_train = t_train
        self.cTG_train = cTG_train
        self.idx = idx
        
        self.lr = LEARNING_RATE
        self.params = list(self.dnn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        
    def pred(self, c_pred, k_pred):
        self.cB = c_pred[:,0].reshape(-1,1)
        self.cDG = c_pred[:,2].reshape(-1,1)
        
        with torch.no_grad():
            for i, p in enumerate(self.dnn.parameters()):
                if i in range(6):
                    p.data.fill_(k_pred[i])
        
    def loss(self, x, y):
        g = x.clone()
        g.requires_grad = True
        
        self.cTG = self.dnn(g)
        
        grad_cTG = autograd.grad(self.cTG, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]
        loss_cTG_ode = self.loss_function(grad_cTG + self.k1*self.cTG - self.k2*self.cDG*self.cB, self.f_hat)
        
        loss_cTG_data = self.loss_function(self.cTG[self.idx], y[self.idx])
        
        loss = self.alpha*loss_cTG_ode + (1-self.alpha)*loss_cTG_data
        
        return loss
    
    def closure(self):
        self.optimizer.zero_grad()
        
        self.pred(self.c_pred, self.k_pred)
        
        loss = self.loss(self.t_train, self.cTG_train)
        
        loss.backward()
        
        return loss

class cDGNN():
    
    def __init__(self, device, f_hat, c_pred, k_pred, t_train, cDG_train, idx):
        self.dnn = subDNN().to(device)
        self.loss_function = nn.MSELoss(reduction = 'mean')

        self.k1 = torch.tensor([k_pred[0]], requires_grad=True).float().to(device)
        self.k2 = torch.tensor([k_pred[1]], requires_grad=True).float().to(device)
        self.k3 = torch.tensor([k_pred[2]], requires_grad=True).float().to(device)
        self.k4 = torch.tensor([k_pred[3]], requires_grad=True).float().to(device)
        self.k5 = torch.tensor([k_pred[4]], requires_grad=True).float().to(device)
        self.k6 = torch.tensor([k_pred[5]], requires_grad=True).float().to(device)
        
        self.k1 = nn.Parameter(self.k1)
        self.k2 = nn.Parameter(self.k2)
        self.k3 = nn.Parameter(self.k3)
        self.k4 = nn.Parameter(self.k4)
        self.k5 = nn.Parameter(self.k5)
        self.k6 = nn.Parameter(self.k6)
        
        self.dnn.register_parameter('k1', self.k1)
        self.dnn.register_parameter('k2', self.k2)
        self.dnn.register_parameter('k3', self.k3)
        self.dnn.register_parameter('k4', self.k4)
        self.dnn.register_parameter('k5', self.k5)
        self.dnn.register_parameter('k6', self.k6)
        
        self.device = device
        self.alpha = ALPHA
        self.f_hat = f_hat
        self.c_pred = c_pred
        self.k_pred = k_pred
        self.t_train = t_train
        self.cDG_train = cDG_train
        self.idx = idx
        
        self.lr = LEARNING_RATE
        self.params = list(self.dnn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        
    def pred(self, c_pred, k_pred):
        self.cB = c_pred[:,0].reshape(-1,1)
        self.cTG = c_pred[:,1].reshape(-1,1)
        self.cMG = c_pred[:,3].reshape(-1,1)
        
        with torch.no_grad():
            for i, p in enumerate(self.dnn.parameters()):
                if i in range(6):
                    p.data.fill_(k_pred[i])
        
    def loss(self, x, y):
        g = x.clone()
        g.requires_grad = True
        
        self.cDG = self.dnn(g)
        
        grad_cDG = autograd.grad(self.cDG, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]
        loss_cDG_ode = self.loss_function(grad_cDG - self.k1*self.cTG + self.k2*self.cDG*self.cB \
                                                   + self.k3*self.cDG - self.k4*self.cMG*self.cB, self.f_hat)
        
        loss_cDG_data = self.loss_function(self.cDG[self.idx], y[self.idx])
        
        loss = self.alpha*loss_cDG_ode + (1-self.alpha)*loss_cDG_data
        
        return loss
    
    def closure(self):
        self.optimizer.zero_grad()
        
        self.pred(self.c_pred, self.k_pred)
        
        loss = self.loss(self.t_train, self.cDG_train)
        
        loss.backward()
        
        return loss
    
class cMGNN():
    
    def __init__(self, device, f_hat, c_pred, k_pred, t_train, cMG_train, idx):
        self.dnn = subDNN().to(device)
        self.loss_function = nn.MSELoss(reduction = 'mean')
        
        self.k1 = torch.tensor([k_pred[0]], requires_grad=True).float().to(device)
        self.k2 = torch.tensor([k_pred[1]], requires_grad=True).float().to(device)
        self.k3 = torch.tensor([k_pred[2]], requires_grad=True).float().to(device)
        self.k4 = torch.tensor([k_pred[3]], requires_grad=True).float().to(device)
        self.k5 = torch.tensor([k_pred[4]], requires_grad=True).float().to(device)
        self.k6 = torch.tensor([k_pred[5]], requires_grad=True).float().to(device)
        
        self.k1 = nn.Parameter(self.k1)
        self.k2 = nn.Parameter(self.k2)
        self.k3 = nn.Parameter(self.k3)
        self.k4 = nn.Parameter(self.k4)
        self.k5 = nn.Parameter(self.k5)
        self.k6 = nn.Parameter(self.k6)
        
        self.dnn.register_parameter('k1', self.k1)
        self.dnn.register_parameter('k2', self.k2)
        self.dnn.register_parameter('k3', self.k3)
        self.dnn.register_parameter('k4', self.k4)
        self.dnn.register_parameter('k5', self.k5)
        self.dnn.register_parameter('k6', self.k6)
        
        self.device = device
        self.alpha = ALPHA
        self.f_hat = f_hat
        self.c_pred = c_pred
        self.k_pred = k_pred
        self.t_train = t_train
        self.cMG_train = cMG_train
        self.idx = idx
        
        self.lr = LEARNING_RATE
        self.params = list(self.dnn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        
    def pred(self, c_pred, k_pred):
        self.cB = c_pred[:,0].reshape(-1,1)
        self.cDG = c_pred[:,2].reshape(-1,1)
        self.cG = c_pred[:,4].reshape(-1,1)
        
        with torch.no_grad():
            for i, p in enumerate(self.dnn.parameters()):
                if i in range(6):
                    p.data.fill_(k_pred[i])
        
    def loss(self, x, y):
        g = x.clone()
        g.requires_grad = True
        
        self.cMG = self.dnn(g)
        
        grad_cMG = autograd.grad(self.cMG, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]
        loss_cMG_ode = self.loss_function(grad_cMG - self.k3*self.cDG + self.k4*self.cMG*self.cB \
                                                   + self.k5*self.cMG - self.k6*self.cG*self.cB, self.f_hat)
        
        loss_cMG_data = self.loss_function(self.cMG[self.idx], y[self.idx])
        
        loss = self.alpha*loss_cMG_ode + (1-self.alpha)*loss_cMG_data
        
        return loss
    
    def closure(self):
        self.optimizer.zero_grad()
        
        self.pred(self.c_pred, self.k_pred)
        
        loss = self.loss(self.t_train, self.cMG_train)
        
        loss.backward()
        
        return loss

class cBNN():
    
    def __init__(self, device, f_hat, c_pred, k_pred, t_train):
        self.dnn = subDNN().to(device)
        self.loss_function = nn.MSELoss(reduction = 'mean')
        
        self.k1 = torch.tensor([k_pred[0]], requires_grad=True).float().to(device)
        self.k2 = torch.tensor([k_pred[1]], requires_grad=True).float().to(device)
        self.k3 = torch.tensor([k_pred[2]], requires_grad=True).float().to(device)
        self.k4 = torch.tensor([k_pred[3]], requires_grad=True).float().to(device)
        self.k5 = torch.tensor([k_pred[4]], requires_grad=True).float().to(device)
        self.k6 = torch.tensor([k_pred[5]], requires_grad=True).float().to(device)
        
        self.k1 = nn.Parameter(self.k1)
        self.k2 = nn.Parameter(self.k2)
        self.k3 = nn.Parameter(self.k3)
        self.k4 = nn.Parameter(self.k4)
        self.k5 = nn.Parameter(self.k5)
        self.k6 = nn.Parameter(self.k6)
        
        self.dnn.register_parameter('k1', self.k1)
        self.dnn.register_parameter('k2', self.k2)
        self.dnn.register_parameter('k3', self.k3)
        self.dnn.register_parameter('k4', self.k4)
        self.dnn.register_parameter('k5', self.k5)
        self.dnn.register_parameter('k6', self.k6)
        
        self.device = device
        self.alpha = ALPHA
        self.f_hat = f_hat
        self.c_pred = c_pred
        self.k_pred = k_pred
        self.t_train = t_train
        
        self.lr = LEARNING_RATE
        self.params = list(self.dnn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        
    def pred(self, c_pred, k_pred):
        self.cTG = c_pred[:,1].reshape(-1,1)
        self.cDG = c_pred[:,2].reshape(-1,1)
        self.cMG = c_pred[:,3].reshape(-1,1)
        self.cG = c_pred[:,4].reshape(-1,1)
        
        with torch.no_grad():
            for i, p in enumerate(self.dnn.parameters()):
                if i in range(6):
                    p.data.fill_(k_pred[i])
        
    def loss(self, x):
        g = x.clone()
        g.requires_grad = True
        
        self.cB = self.dnn(g)
        
        grad_cB = autograd.grad(self.cB, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]
        loss_cB_ode = self.loss_function(grad_cB - self.k1*self.cTG + self.k2*self.cDG*self.cB \
                                                 - self.k3*self.cDG + self.k4*self.cMG*self.cB \
                                                 - self.k5*self.cMG + self.k6*self.cG*self.cB, self.f_hat)
        
        loss_cB_data = self.loss_function(self.cB[0], self.f_hat[0])
        
        loss = self.alpha*loss_cB_ode + (1-self.alpha)*loss_cB_data
        
        return loss
    
    def closure(self):
        self.optimizer.zero_grad()
        
        self.pred(self.c_pred, self.k_pred)
        
        loss = self.loss(self.t_train)
        
        loss.backward()
        
        return loss

class cGNN():
    
    def __init__(self, device, f_hat, c_pred, k_pred, t_train):
        self.dnn = subDNN().to(device)
        self.loss_function = nn.MSELoss(reduction = 'mean')
        
        self.k1 = torch.tensor([k_pred[0]], requires_grad=True).float().to(device)
        self.k2 = torch.tensor([k_pred[1]], requires_grad=True).float().to(device)
        self.k3 = torch.tensor([k_pred[2]], requires_grad=True).float().to(device)
        self.k4 = torch.tensor([k_pred[3]], requires_grad=True).float().to(device)
        self.k5 = torch.tensor([k_pred[4]], requires_grad=True).float().to(device)
        self.k6 = torch.tensor([k_pred[5]], requires_grad=True).float().to(device)
        
        self.k1 = nn.Parameter(self.k1)
        self.k2 = nn.Parameter(self.k2)
        self.k3 = nn.Parameter(self.k3)
        self.k4 = nn.Parameter(self.k4)
        self.k5 = nn.Parameter(self.k5)
        self.k6 = nn.Parameter(self.k6)
        
        self.dnn.register_parameter('k1', self.k1)
        self.dnn.register_parameter('k2', self.k2)
        self.dnn.register_parameter('k3', self.k3)
        self.dnn.register_parameter('k4', self.k4)
        self.dnn.register_parameter('k5', self.k5)
        self.dnn.register_parameter('k6', self.k6)
        
        self.device = device
        self.alpha = ALPHA
        self.f_hat = f_hat
        self.c_pred = c_pred
        self.k_pred = k_pred
        self.t_train = t_train
        
        self.lr = LEARNING_RATE
        self.params = list(self.dnn.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=self.lr)
        
    def pred(self, c_pred, k_pred):
        self.cB = c_pred[:,0].reshape(-1,1)
        self.cMG = c_pred[:,3].reshape(-1,1)
        
        with torch.no_grad():
            for i, p in enumerate(self.dnn.parameters()):
                if i in range(6):
                    p.data.fill_(k_pred[i])
        
    def loss(self, x):
        g = x.clone()
        g.requires_grad = True
        
        self.cG = self.dnn(g)
        
        grad_cG = autograd.grad(self.cG, g, torch.ones(x.shape[0], 1).to(self.device), retain_graph=True, create_graph=True)[0]
        loss_cG_ode = self.loss_function(grad_cG - self.k5*self.cMG + self.k6*self.cG*self.cB, self.f_hat)
        
        loss_cG_data = self.loss_function(self.cG[0], self.f_hat[0])
        
        loss = self.alpha*loss_cG_ode + (1-self.alpha)*loss_cG_data
        
        return loss
    
    def closure(self):
        self.optimizer.zero_grad()
        
        self.pred(self.c_pred, self.k_pred)
        
        loss = self.loss(self.t_train)
        
        loss.backward()
        
        return loss

def train_cNN(cNN, cNN_index, c_pred, k_pred, t_train):
    
    cNN.optimizer.step(cNN.closure)
    for i, p in enumerate(cNN.dnn.parameters()):
        if i in range(6):
            p.data.clamp_(min=0.)
            k_pred[i] = float(p.detach().numpy())
    
    c_pred[:,cNN_index] = cNN.dnn(t_train).detach().clone().flatten()
