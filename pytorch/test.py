import torch.nn as nn
import numpy as np
import pandas as pd
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('bio.csv', sep=',')
t = data['t'].to_numpy().reshape(-1,1)
x = torch.from_numpy(t).float().to(device)

layers = np.array([1,10,10,10,5])

activation = nn.Tanh()

linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

for i in range(len(layers)-1):
            
            nn.init.xavier_normal_(linears[i].weight.data, gain=1.0)
            
            nn.init.zeros_(linears[i].bias.data)

a = x.float()
        
for i in range(len(layers)-2):
    
    z = linears[i](a)
                
    a = activation(z)
    
a = linears[-1](a)

loss_function = nn.MSELoss(reduction = 'mean')
loss_function