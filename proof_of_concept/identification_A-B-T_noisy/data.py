# Libraries importation
import numpy as np
import torch

def EDO(y, prm):

    k1 = prm.k1
    k2 = prm.k2
    dHrx = prm.dHrx
    Q = prm.Q
    V = prm.V
    e = prm.e
    c1 = prm.c1
    c2 = prm.c2
    m_Cp = prm.m_Cp

    f = np.zeros(3)

    cA = y[0]
    cB = y[1]
    T = y[2]

    f[0] = -k1 * cA + k2 * cB
    f[1] = -k2 * cB + k1 * cA
    f[2] = (e*Q + dHrx * f[0] * V + c1*T + c2) / m_Cp

    return f

def euler(y0, t, prm):

    mat_y = np.array([y0])
    
    prm.k1 = prm.A1*np.exp(-prm.E1/y0[2])
    prm.k2 = prm.A2*np.exp(-prm.E2/y0[2])

    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        y = y0 + dt * EDO(y0, prm)

        mat_y = np.append(mat_y, [y], axis=0)

        y0 = np.copy(y)

        prm.k1 = prm.A1*np.exp(-prm.E1/y[2])
        prm.k2 = prm.A2*np.exp(-prm.E2/y[2])

    return t, mat_y

def create_data(y0, t1, t2, collocation_points, prm, error_percentage):

    t = np.linspace(t1, t2, collocation_points)

    _, y_num = euler(y0, t, prm)

    y_num_no_noise = np.copy(y_num)

    # Add noise
    n_cA = np.random.normal(y_num[:,0], error_percentage*y_num[:,0])
    n_cB = np.random.normal(y_num[:,1], error_percentage*y_num[:,1])
    n_T = np.random.normal(y_num[:,2], 0.01*y_num[:,2])
    y_num[:,0] = n_cA
    y_num[:,1] = n_cB
    y_num[:,2] = n_T

    return t.reshape(-1,1), y_num, y_num_no_noise

def put_in_device(x, y, device):

    X = torch.from_numpy(x).float().to(device)
    Y = torch.from_numpy(y).float().to(device)

    return X, Y
