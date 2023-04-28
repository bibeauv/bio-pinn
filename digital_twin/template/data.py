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

def put_in_device(x, y, z, device):

    X = torch.from_numpy(x).float().to(device)
    Y = torch.from_numpy(y).float().to(device)
    Z = torch.from_numpy(z).float().to(device)

    return X, Y, Z

def gather_data(files, T_file):
    
    C = read_data(files[0])
    T_df = pd.read_csv(T_file)
    P = float(files[0].split('_')[1].split('W')[0])
    t = T_df[T_df['Power'] == P]['Time'].to_numpy().reshape(-1,1)
    T = T_df[T_df['Power'] == P]['Temperature'].to_numpy().reshape(-1,1)
    Q = T_df[T_df['Power'] == P]['Power'].to_numpy().reshape(-1,1)

    idx = find_idx(t, C)
    idx_y0 = [0]
    idx_yf = []

    X = np.concatenate((t, T), axis=1)
    Y = np.copy(C[:,1:])
    Z = np.copy(T)

    len_t = len(t)
    for i in range(1,len(files)):
        new_C = read_data(files[i])
        P = float(files[i].split('_')[1].split('W')[0])
        new_t = T_df[T_df['Power'] == P]['Time'].to_numpy().reshape(-1,1)
        new_Q = T_df[T_df['Power'] == P]['Power'].to_numpy().reshape(-1,1)
        new_T = T_df[T_df['Power'] == P]['Temperature'].to_numpy().reshape(-1,1)

        new_idx = find_idx(new_t, new_C)

        new_X = np.concatenate((new_t, new_T), axis=1)
        X = np.concatenate((X, new_X), axis=0)
        Y = np.concatenate((Y, new_C[:,1:]), axis=0)
        Z = np.concatenate((Z, new_T), axis=0)

        for j in range(len(new_idx)):
            new_idx[j] = new_idx[j] + len_t
        idx = idx + new_idx
        idx_y0 = idx_y0 + [len_t]
        if P == 4.0:
            idx_yf = idx_yf + [new_idx[-1]]
        len_t += len(new_t)
        
    return X, Y, Z, idx, idx_y0, idx_yf