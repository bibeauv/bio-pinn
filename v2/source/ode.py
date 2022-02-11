import numpy as np

class ODE:
    
    def __init__(self, eq):
        self.f = list(np.zeros(eq))
        self.k = []
        self.C = []
        self.y = []
        self.X = []
        self.eq = eq
    
    def ode_function(self):
        if self.eq == 2:
            self.f[0] = self.k[0]*self.C[0]
            self.f[1] = self.k[1]*self.C[1]
        if self.eq == 1:
            self.f[0] = self.k[0]*self.C[0]
    
    def rate_function(self, T, A, E):
        self.k.append(A*np.exp(-E/T))
        
    def rate(self, k):
        self.k = k
          
    def euler_explicite(self, T, A, E, dt, tf, y0):
        for i in range(0, len(T)):
            self.rate_function(T[i], A, E)
            for j in range(0, len(tf)):
                t = 0
                self.C = [y0, 0]
                while t < tf[j]:
                    self.ode_function()
                    CA = self.C[0] + dt*-self.f[0]
                    CB = self.C[1] + dt*self.f[0]
                    self.C = [CA, CB]
                    t = t + dt
                self.X.append([T[i], tf[j]])
                self.y.append(self.C)
        return self.y, self.X
    
    def euler_explicite_multi(self, k, dt, tf, y0):
        self.rate(k)
        for j in range(0, len(tf)):
            t = 0
            self.C = [y0, 0, 0]
            while t < tf[j]:
                self.ode_function()
                CA = self.C[0] + dt*-self.f[0]
                CB = self.C[1] + dt*(self.f[0] - self.f[1])
                CC = self.C[2] + dt*self.f[1]
                self.C = [CA, CB, CC]
                t = t + dt
            self.y.append(self.C)
            self.X.append(tf[j])
        return self.y, self.X
    
    def analytical_solution(self, T, A, E, tf, y0):
        self.yA = []
        for temperature in T:
            self.rate_function(temperature, A, E)
            for t in tf:
                self.yA.append(y0*np.exp(-self.k[0]*t))
        return self.yA
    
    def analytical_solution_multi(self, k, tf, y0):
        self.rate(k)
        for t in tf:
            self.yA.append(y0*np.exp(-self.k*t))
        return self.y