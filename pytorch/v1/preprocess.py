# Importation de librairies
import torch

import numpy as np
import pandas as pd

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# Read data
def read_data(file, device, t1, t2, points):
    data = pd.read_csv(file, sep=',')
    data = data.replace(np.nan, 0.)
    
    t = np.linspace(t1, t2, num=points)
    
    idx = []
    for ti in data['t']:
        idx.append(np.where(t == ti)[0][0])
    
    t = t.reshape(-1,1)
    
    cB = np.zeros(t.shape[0])
    cB[idx] = data['cB'].to_numpy()
    cB = cB.reshape(-1,1)
    
    cTG = np.zeros(t.shape[0])
    cTG[idx] = data['cTG'].to_numpy()
    cTG = cTG.reshape(-1,1)
    
    cDG = np.zeros(t.shape[0])
    cDG[idx] = data['cDG'].to_numpy()
    cDG = cDG.reshape(-1,1)
    
    cMG = np.zeros(t.shape[0])
    cMG[idx] = data['cMG'].to_numpy()
    cMG = cMG.reshape(-1,1)
    
    cG = np.zeros(t.shape[0])
    cG[idx] = data['cG'].to_numpy()
    cG = cG.reshape(-1,1)
    
    t_train = torch.from_numpy(t).float().to(device)
    cB_train = torch.from_numpy(cB).float().to(device)
    cTG_train = torch.from_numpy(cTG).float().to(device)
    cDG_train = torch.from_numpy(cDG).float().to(device)
    cMG_train = torch.from_numpy(cMG).float().to(device)
    cG_train = torch.from_numpy(cG).float().to(device)
    
    c_pred = torch.cat((cB_train, cTG_train, cDG_train, cMG_train, cG_train), 1)
    c_pred = c_pred.to(device)
    
    return t_train, cB_train, cTG_train, cDG_train, cMG_train, cG_train, c_pred, idx
