# Libraries importation
import numpy as np
import torch

# Set seed
np.random.seed(1234)

def EDO(y, prm):

    k1 = prm.k1
    dHrx = prm.dHrx
    Q = prm.Q
    V = prm.V
    e = prm.e
    c1 = prm.c1
    c2 = prm.c2
    m_Cp = prm.m_Cp

    f = np.zeros(2)

    cA = y[0]
    T = y[1]

    f[0] = -k1 * cA
    f[1] = (e*Q + dHrx * f[0] * V + c1*T + c2) / m_Cp

    return f

def euler(y0, t, prm):

    mat_y = np.array([y0])
    
    prm.k1 = prm.A*np.exp(-prm.E/y0[1])

    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        y = y0 + dt * EDO(y0, prm)

        mat_y = np.append(mat_y, [y], axis=0)

        y0 = np.copy(y)

        prm.k1 = prm.A*np.exp(-prm.E/y[1])

    return t, mat_y

def create_data(y0, t1, t2, collocation_points, prm):

    t = np.linspace(t1, t2, collocation_points)

    _, y_num = euler(y0, t, prm)

    return t.reshape(-1,1), y_num

def put_in_device(x, y, device):

    X = torch.from_numpy(x).float().to(device)
    Y = torch.from_numpy(y).float().to(device)

    return X, Y
