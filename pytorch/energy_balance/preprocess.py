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
def read_data(Ea, A, t1, t2, T1, T2, points, device, prm):
    data = {}
    data['t'] = np.linspace(t1, t2, 6)
    
    t = np.linspace(t1, t2, num=points)
    
    dt = t2 / (points - 1)
    t_num, y_num = euler_explicite([0.6, 25.], Ea, A, dt, t2, prm)
    
    idx = []
    for ti in data['t']:
        idx.append(np.where(t == ti)[0][0])
    
    t = t.reshape(-1,1)
    
    cTG = np.zeros(t.shape[0])
    cTG[idx] = y_num[idx,0].flatten()
    cTG = cTG.reshape(-1,1)

    T = np.zeros(t.shape[0])
    T[idx] = y_num[idx,1].flatten()
    T = T.reshape(-1,1)
    
    t_train = torch.from_numpy(t).float().to(device)
    T_train = torch.from_numpy(T).float().to(device)
    cTG_train = torch.from_numpy(cTG).float().to(device)
    
    y_pred = {}
    y_pred['cTG'] = cTG_train
    y_pred['Temp'] = T_train
    
    return t_train, T_train, cTG_train, y_pred, idx
