# Importation de librairies
import torch

import numpy as np
import pandas as pd

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# Read data
def read_data(file):
    data = pd.read_csv(file, sep=',')
    data = data.replace(np.nan, 0.)
    
    C = data.to_numpy()

    return C

def find_idx(t, C):
    idx = []
    t_data = C[:,0]
    for ti in t_data:
        idx.append(np.where(t == ti)[0][0])
        
    return idx

def put_in_device(x, y, device):

    X = torch.from_numpy(x).float().to(device)
    Y = torch.from_numpy(y).float().to(device)

    return X, Y

def gather_data(files):
    
    C = read_data(files[0])
    t = C[:,0].reshape(-1,1)
    Q = C[:,1].reshape(-1,1)
    T = C[:,2].reshape(-1,1)

    idx = find_idx(t, C)
    idx_y0 = [0]

    X = np.concatenate((t, Q), axis=1)
    Y = np.copy(T).reshape(-1,1)
        
    return X, Y, idx, idx_y0