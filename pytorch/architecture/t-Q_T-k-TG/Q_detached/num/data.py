# Importation de librairies
import numpy as np
import torch

# Set seed
np.random.seed(1234)

def EDO(y, prm):

    k = prm.k
    dHrx = prm.dHrx
    Q = prm.Q
    V = prm.V

    f = np.zeros(2)

    f[0] = -k * y[0]
    f[1] = Q + dHrx * f[0] * V

    return f

def euler(y0, dt, tf, prm):

    mat_y = np.array([y0])
    t = np.array([0])
    
    while t[-1] < tf-dt:
        y = y0 + dt * EDO(y0, prm)

        mat_y = np.append(mat_y, [y], axis=0)
        t = np.append(t, t[-1]+dt)

        y0 = np.copy(y)

        prm.k = prm.A*np.exp(-prm.Ea/y[1])

    return t, mat_y

def create_data(y0, t1, t2, collocation_points, prm):

    t = np.linspace(t1, t2, collocation_points)
    dt = t2 / (collocation_points - 1)

    _, y_num = euler(y0, dt, t2, prm)

    return t.reshape(-1,1), y_num

def put_in_device(x, y, device):

    X = torch.from_numpy(x).float().to(device)
    Y = torch.from_numpy(y).float().to(device)

    return X, Y

def find_idx(t_data, t_collocation):

    idx = []
    for t in t_data:
        idx.extend(np.where(t == t_collocation)[0].tolist())

    return idx