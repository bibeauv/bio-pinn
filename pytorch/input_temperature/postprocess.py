# Importation de librairies
import numpy as np
import matplotlib.pyplot as plt

# Set seed
np.random.seed(1234)

def EDO(y, prm):
    
    k1 = prm[0]
    
    f = np.zeros(1)
    
    f[0] = - k1*y[0]
    
    return f

def euler_explicite(y0, T, Ea, A, dt, tf, prm):
    
    mat_y = np.array([y0])
    
    j = 1
    t = np.array([0])
    while t[-1] < tf-dt:
        y = y0 + dt * EDO(y0, prm)
        
        mat_y = np.append(mat_y, [y], axis=0)
        t = np.append(t, t[-1]+dt)
        
        y0 = np.copy(y)
        
        prm = [A*np.exp(-Ea/T[j])]
        j += 1
    
    return t, mat_y
