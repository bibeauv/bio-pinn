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
def read_data(Ea, A, Q, t1, t2, points, device, prm):
    data = {}
    data['t'] = np.linspace(t1, t2, 6)
    
    t = np.linspace(t1, t2, num=points)
    
    dt = t2 / (points - 1)
    y_conc = np.array([])
    y_temp = np.array([])
    t_ = np.array([])
    for q in Q:
        prm[1] = q
        _, y_num = euler_explicite([0.6, 35.], Ea, A, dt, t2, prm)
        y_conc = np.append(y_conc, y_num[:,0])
        y_temp = np.append(y_temp, y_num[:,1].flatten())
        t_ = np.append(t_, t.flatten())
    
    Q_train = np.zeros(t_.shape[0])
    j = 0
    for i in range(0,len(t_),points):
        Q_train[i:i+points] = Q[j]
        j += 1
    Q_train = Q_train.reshape(-1,1)

    idx = []
    for ti in data['t']:
        idx.extend(np.where(t_ == ti)[0].tolist())
    
    t = t_.reshape(-1,1)
    
    cTG = np.zeros(t_.shape[0])
    cTG[idx] = y_conc[idx].flatten()
    cTG = cTG.reshape(-1,1)

    T = np.zeros(t_.shape[0])
    T[idx] = y_temp[idx].flatten()
    T = T.reshape(-1,1)
    
    t_train = torch.from_numpy(t).float().to(device)
    Q_train = torch.from_numpy(Q_train).float().to(device)
    T_train = torch.from_numpy(T).float().to(device)
    cTG_train = torch.from_numpy(cTG).float().to(device)
    
    y_pred = {}
    y_pred['cTG'] = cTG_train
    y_pred['Temp'] = T_train
    
    return t_train, Q_train, T_train, cTG_train, y_pred, idx
