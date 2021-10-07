import numpy as np

class ODE:
    
    def __init__(self):
        self.f1 = 0
        self.f2 = 0
        self.k = 0
        self.CA = 0
        self.CB = 0
        self.y = []
        self.x1 = []
        self.x2 = []
    
    def ode_function(self):
        self.f1 = self.k*self.CA
        self.f2 = -self.k*self.CA
    
    def rate_function(self, T, A, E):
        R = 8.3145
        self.k = A*np.exp(-E/R/T)
    
    def euler_explicite(self, T, A, E, dt, tf, y0):
        for i in range(0, len(T)):
            self.rate_function(T[i], A, E)
            for j in range(0, len(tf)):
                t = 0
                self.CA = y0
                self.CB = 0
                while t < tf[j]:
                    self.ode_function()
                    self.CA = self.CA + dt*self.f2
                    self.CB = self.CB + dt*self.f1
                    t = t + dt
                self.x1.append(T[i])
                self.x2.append(tf[j])
                self.y.append(self.CB)
        return self.y, self.x1, self.x2