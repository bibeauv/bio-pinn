from scipy.optimize import fsolve
import math

def equations(p):
    A, E = p
    R = 8.314
    T0 = 35+273.15
    f1 = A * math.exp(-E / R / T0) - 0.0096950214356184
    f2 = A * math.exp(-E / R / T0) * (-E / R / (T0**2)) - 0.00612987345084548
    return (f1, f2)

A, E = fsolve(equations, (1,1))
print(A, E)
print(equations((A,E)))