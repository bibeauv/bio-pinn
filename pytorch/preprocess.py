# Importation de librairies
import torch

import numpy as np
import pandas as pd

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# Read data
def read_data(file, device):
    data = pd.read_csv(file, sep=',')
    data = data.replace(np.nan, 0.)
    
    t = data['t'].to_numpy().reshape(-1,1)
    cB = data['cB'].to_numpy().reshape(-1,1)
    cTG = data['cTG'].to_numpy().reshape(-1,1)
    cDG = data['cDG'].to_numpy().reshape(-1,1)
    cMG = data['cMG'].to_numpy().reshape(-1,1)
    cG = data['cG'].to_numpy().reshape(-1,1)
    
    t_train = torch.from_numpy(t).float().to(device)
    cB_train = torch.from_numpy(cB).float().to(device)
    cTG_train = torch.from_numpy(cTG).float().to(device)
    cDG_train = torch.from_numpy(cDG).float().to(device)
    cMG_train = torch.from_numpy(cMG).float().to(device)
    cG_train = torch.from_numpy(cG).float().to(device)
    
    c_pred = torch.cat((cB_train, cTG_train, cDG_train, cMG_train, cG_train), 1)
    c_pred = c_pred.to(device)
    
    return t_train, cB_train, cTG_train, cDG_train, cMG_train, cG_train, c_pred
