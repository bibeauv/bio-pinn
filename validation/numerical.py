import numpy as np

def EDO(y, Ti, prm):

    f = np.zeros(5)

    cTG = y[0]
    cDG = y[1]
    cMG = y[2]
    cG = y[3]
    cME = y[4]

    T = prm.T
    k1 = prm.A1 + prm.E1 * (Ti - T[0])
    k2 = prm.A2 + prm.E2 * (Ti - T[0])
    k3 = prm.A3 + prm.E3 * (Ti - T[0])
    k4 = prm.A4 + prm.E4 * (Ti - T[0])
    k5 = prm.A5 + prm.E5 * (Ti - T[0])
    k6 = prm.A6 + prm.E6 * (Ti - T[0])

    f[0] = - k1 * cTG + k2 * cDG * cME
    f[1] = + k1 * cTG - k2 * cDG * cME - k3 * cDG + k4 * cMG * cME
    f[2] = + k3 * cDG - k4 * cMG * cME - k5 * cMG + k6 * cG * cME
    f[3] = + k5 * cMG - k6 * cG * cME
    f[4] = + k1 * cTG - k2 * cDG * cME + k3 * cDG - k4 * cMG * cME + k5 * cMG - k6 * cG * cME

    return f

def euler(y0, t, prm):

    mat_y = np.array([y0])

    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        Ti = prm.T[i]
        y = y0 + dt * EDO(y0, Ti, prm)

        mat_y = np.append(mat_y, [y], axis=0)

        y0 = np.copy(y)

    return t, mat_y