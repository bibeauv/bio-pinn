# Importation de librairies
import torch

import numpy as np
import pandas as pd
import scipy

# Set seed
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# Read data

def read_microwave_temperature(file):

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

    # Filter
    sos = scipy.signal.butter(2, 1/20, btype='lowpass', analog=False, output='sos')
    Temp_ref = Temp - np.min(Temp)
    T_filtered = scipy.signal.sosfilt(sos, Temp_ref)
    T_filtered = T_filtered + np.min(Temp)

    return t, T_filtered, Q

def read_data(GC_file, MW_file, device):

    data = pd.read_csv(GC_file, sep=',')
    data = data.replace(np.nan, 0.)
    
    t, T, Q = read_microwave_temperature(MW_file)

    t = t.reshape(-1,1)
    T = T.reshape(-1,1)
    Q = Q.reshape(-1,1)
    
    idx = []
    for ti in data['t']:
        idx.append(np.where(t == ti)[0][0])
    
    X = np.concatenate((t, T), axis=1)
    X_train = torch.from_numpy(X).float().to(device)

    Y = np.zeros((len(t), 7))
    for i in range(5):
        Y[idx,i] = data.iloc[:,i+1].to_numpy()
    Y[:,5] = T.flatten()
    Y[:,6] = Q.flatten()
    Y_train = torch.from_numpy(Y).float().to(device)

    return X_train, Y_train, idx
