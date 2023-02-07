# Importation de librairies
import torch

import numpy as np
import pandas as pd

from postprocess import *

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# Read data
def read_data(Ea, A, t1, t2, T1, T2, points, device):
    data = {}
    data['t'] = np.linspace(t1, t2, 6)
    
    t = np.linspace(t1, t2, num=points)
    T = np.linspace(T1, T2, num=points)
    
    dt = t2 / (points - 1)
    t_num, y_num = euler_explicite([0.6], T, Ea, A, dt, t2, [A*np.exp(-Ea/T[0])])
    
    idx = []
    for ti in data['t']:
        idx.append(np.where(t == ti)[0][0])
    
    t = t.reshape(-1,1)
    T = T.reshape(-1,1)
    
    cTG = np.zeros(t.shape[0])
    cTG[idx] = y_num[idx].flatten()
    cTG = cTG.reshape(-1,1)
    
    t_train = torch.from_numpy(t).float().to(device)
    T_train = torch.from_numpy(T).float().to(device)
    cTG_train = torch.from_numpy(cTG).float().to(device)
    
    c_pred = torch.cat((cTG_train,), 1)
    c_pred = c_pred.to(device)
    
    return t_train, T_train, cTG_train, c_pred, idx
