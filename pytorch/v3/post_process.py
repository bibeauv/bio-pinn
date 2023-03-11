import numpy as np

def edo(y, k, Q):

    dHrx = -45650
    e = 0.1
    m = 5.2
    Cp = 1.7
    V = 6.3
    pertes = 0.2

    TG = y[0]
    T = y[1]

    f = np.zeros(2)

    f[0] = -k * TG
    f[1] = (1/(m*Cp))*(e*Q + dHrx*f[0]*V/1000 - pertes)

    return f

def euler_explicite(y0, t, k, Q):

    mat_y = np.array([y0])

    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        Q_i = Q[i]
        k_i = k[i]
        y0 = y0 + dt * edo(y0, k_i, Q_i)
        mat_y = np.append(mat_y, [y0], axis=0)

    return mat_y