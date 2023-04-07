# Importation de librairies
import torch

import numpy as np
import pandas as pd

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# Read data
def read_GC_data(file):
    data = pd.read_csv(file, sep=',')
    data = data.replace(np.nan, 0.)
    
    C = data.to_numpy()

    return C

def read_MW_data(file):
    with open(file) as f:
        lines = f.readlines()

    count = 0
    for line in lines:
        line_content = line.split(';')
        if line_content[0] == 'hh:mm:ss':
            start = count
            break
        count += 1
    start += 1

    t = []
    T_ir = []
    P = []
    for i in range(start,len(lines)-1):
        time = lines[i].split(';')[0].split(':')
        minutes = float(time[1])
        seconds = float(time[2])
        t.append(minutes*60+seconds)
        T_ir.append(float(lines[i].split(';')[2]))
        P.append(float(lines[i].split(';')[5]))

    t = np.array(t)
    Temp = np.array(T_ir)
    Q = np.array(P)

    return t, Temp, Q

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

def gather_data(GC_files, MW_files):
    
    C = read_GC_data(GC_files[0])
    t, T, Q = read_MW_data(MW_files[0])
    idx = find_idx(t, C)
    X = np.copy(t).reshape(-1,1)
    Y = np.concatenate((C[:,1:], T.reshape(-1,1)), axis=1)
    
    for i in range(1,len(GC_files)):
        new_C = read_GC_data(GC_files[i])
        new_t, new_T, new_Q = read_MW_data(MW_files[i])
        new_idx = find_idx(new_t, new_C)
        X = np.concatenate((X, new_t.reshape(-1,1)), axis=0)
        Y = np.concatenate((Y, np.concatenate((new_C[:,1:], new_T.reshape(-1,1)), axis=1)), axis=0)
        for j in range(len(new_idx)):
            new_idx[j] = new_idx[j] + len(t)
        idx = idx + new_idx
        t = np.copy(new_t)
        
    return X, Y, idx