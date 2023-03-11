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
    X_train = torch.from_numpy(t).float().to(device)

    Y = np.zeros((len(t), len(data.columns)-1))
    for i in range(Y.shape[1]):
        Y[idx,i] = data.iloc[:,i+1].to_numpy()
    Y_train = torch.from_numpy(Y).float().to(device)

    return X_train, Y_train, idx
