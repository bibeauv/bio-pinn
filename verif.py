# Classes
import sys
import os
sys.path.append(os.getcwd() + '\source')
from source.ode import ODE
# PINN
import sciann as sn
# Other
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Generate the artificial data
ode = ODE()
T = [1000]
A = 5
E = 1000
dt = 0.0001
tf = [0.0, 0.1, 0.2, 0.3, 0.4]
y0 = 1
# x1 : Temperature
# x2 : Reaction time
yA_ana = ode.analytical_solution(T, A, E, tf, y0)

# PINN
t = sn.Variable()
cA = sn.Functional("cA", [t], 4*[10], activation="tanh")
# Parameters
k = sn.Parameter(0.0, inputs=[t])
# ODE
L1 = sn.math.diff(cA, t) + k*cA
# Initial condition
TOL = 0.001
IC = (1-sn.utils.sign(t - TOL)) * (cA - 1)
# Supervised learning with data
DATA = cA
# Model
m = sn.SciModel([t],
                [L1, IC, DATA], optimizer="adagrad")
m.train([np.array(tf)],
        ["zeros", "zeros", np.array(yA_ana)],
        epochs=5000,
        learning_rate=0.1)
# Predict
y_pred = cA.eval(m, [np.array(tf)])
print(r2_score(y_pred, yA_ana))
# Visualization
plt.plot(tf, yA_ana)
plt.plot(tf, y_pred, 'o')
plt.legend(['Analytical Solution', 'Predictions'])
plt.xlabel('Time')
plt.ylabel('Concentration of A')
plt.show()