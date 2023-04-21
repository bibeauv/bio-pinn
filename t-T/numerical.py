import numpy as np

def EDO(y, prm):

    f = np.zeros(1)

    T = y[0]

    Q = prm.Q
    e = prm.e
    c1 = prm.c1
    c2 = prm.c2
    m_Cp = prm.m_Cp

    f[0] = (e*Q + + c1*T + c2) / m_Cp

    return f

def euler(y0, t, prm):

    mat_y = np.array([y0])

    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        y = y0 + dt * EDO(y0, prm)

        mat_y = np.append(mat_y, [y], axis=0)

        y0 = np.copy(y)

    return t, mat_y