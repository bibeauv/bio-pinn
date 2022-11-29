# Importation de librairies
import numpy as np

# Set seed
np.random.seed(1234)

def EDO(y, prm):
    
    k1 = prm[0]
    k2 = prm[1]
    k3 = prm[2]
    k4 = prm[3]
    k5 = prm[4]
    k6 = prm[5]
    
    f = np.zeros(5)
    
    f[1] = - k1*y[1] + k2*y[2]*y[0]
    f[2] = + k1*y[1] - k2*y[2]*y[0] - k3*y[2] + k4*y[3]*y[0]
    f[3] = + k3*y[2] - k4*y[3]*y[0] - k5*y[3] + k6*y[4]*y[0]
    f[4] = + k5*y[3] - k6*y[4]*y[0]
    f[0] = + k1*y[1] - k2*y[2]*y[0] + k3*y[2] - k4*y[3]*y[0] + k5*y[3] - k6*y[4]*y[0]
    
    return f

def euler_explicite(y0, dt, tf, prm):
    
    mat_y = np.array([y0])
    
    t = np.array([0])
    while t[-1] < tf:
        y = y0 + dt * EDO(y0, prm)
        
        mat_y = np.append(mat_y, [y], axis=0)
        t = np.append(t, t[-1]+dt)
        
        y0 = np.copy(y)
    
    return t, mat_y
