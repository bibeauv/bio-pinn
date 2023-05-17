import numpy as np

def EDO(y, prm, i):

    f = np.zeros(5)

    cTG = y[0]
    cDG = y[1]
    cMG = y[2]
    cG = y[3]
    cME = y[4]

    T = prm.T
    Ti = T[i]

    k1 = prm.A1 + prm.E1 * (Ti - np.min(T))
    k2 = prm.A2 + prm.E2 * (Ti - np.min(T))
    k3 = prm.A3 + prm.E3 * (Ti - np.min(T))
    k4 = prm.A4 + prm.E4 * (Ti - np.min(T))
    k5 = prm.A5 + prm.E5 * (Ti - np.min(T))
    k6 = prm.A6 + prm.E6 * (Ti - np.min(T))

    # k1 = prm.A1 * np.exp(-prm.E1 / Ti)
    # k2 = prm.A2 * np.exp(-prm.E2 / Ti)
    # k3 = prm.A3 * np.exp(-prm.E3 / Ti)
    # k4 = prm.A4 * np.exp(-prm.E4 / Ti)
    # k5 = prm.A5 * np.exp(-prm.E5 / Ti)
    # k6 = prm.A6 * np.exp(-prm.E6 / Ti)

    f[0] = - k1 * cTG + k2 * cDG * cME
    f[1] = + k1 * cTG - k2 * cDG * cME - k3 * cDG + k4 * cMG * cME
    f[2] = + k3 * cDG - k4 * cMG * cME - k5 * cMG + k6 * cG * cME
    f[3] = + k5 * cMG - k6 * cG * cME
    f[4] = + k1 * cTG - k2 * cDG * cME + k3 * cDG - k4 * cMG * cME + k5 * cMG - k6 * cG * cME
    # f[5] = (e*Q + c1*T + c2) / m_Cp

    return f

def euler(y0, t, prm):

    mat_y = np.array([y0])

    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        y = y0 + dt * EDO(y0, prm, i)

        mat_y = np.append(mat_y, [y], axis=0)

        y0 = np.copy(y)

    return t, mat_y