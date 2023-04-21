import numpy as np

def EDO(y, prm):

    f = np.zeros(6)

    cTG = y[0]
    cDG = y[1]
    cMG = y[2]
    cG = y[3]
    cME = y[4]
    T = y[5]

    k1 = prm.A1 + prm.E1 * (T - 34)
    k2 = prm.A2 + prm.E2 * (T - 34)
    k3 = prm.A3 + prm.E3 * (T - 34)
    k4 = prm.A4 + prm.E4 * (T - 34)
    k5 = prm.A5 + prm.E5 * (T - 34)
    k6 = prm.A6 + prm.E6 * (T - 34)

    Q = prm.Q
    e = prm.e
    c1 = prm.c1
    c2 = prm.c2
    m_Cp = prm.m_Cp

    f[0] = - k1 * cTG + k2 * cDG * cME
    f[1] = + k1 * cTG - k2 * cDG * cME - k3 * cDG + k4 * cMG * cME
    f[2] = + k3 * cDG - k4 * cMG * cME - k5 * cMG + k6 * cG * cME
    f[3] = + k5 * cMG - k6 * cG * cME
    f[4] = + k1 * cTG - k2 * cDG * cME + k3 * cDG - k4 * cMG * cME + k5 * cMG - k6 * cG * cME
    f[5] = (e*Q + c1*T + c2) / m_Cp

    return f

def euler(y0, t, prm):

    mat_y = np.array([y0])

    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        y = y0 + dt * EDO(y0, prm)

        mat_y = np.append(mat_y, [y], axis=0)

        y0 = np.copy(y)

    return t, mat_y