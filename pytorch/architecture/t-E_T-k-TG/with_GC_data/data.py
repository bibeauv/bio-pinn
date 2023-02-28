# Importation de librairies
import numpy as np
import torch

# Set seed
np.random.seed(1234)

def read_data(files):

    t = []
    T_ir = []
    P = []

    for file in files:
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

        for i in range(start,len(lines)-1):
            time = lines[i].split(';')[0].split(':')
            minutes = float(time[1])
            seconds = float(time[2])
            t.append(minutes*60+seconds)
            T_ir.append(float(lines[i].split(';')[2]))
            P.append(float(lines[i].split(';')[5]))

    return t, T_ir, P

def transform_power_to_energy(t, P):

    E = [0.]
    for i in range(1,len(P)):
        Energy = (P[i] + P[i-1])*(t[i] - t[i-1])/2
        E.append(Energy+E[-1])

    return E

def put_in_device(x, y, device):

    X = torch.from_numpy(x).float().to(device)
    Y = torch.from_numpy(y).float().to(device)

    return X, Y

def find_idx(t_data, t_collocation):

    idx = []
    for t in t_data:
        idx.extend(np.where(t == t_collocation)[0].tolist())

    return idx