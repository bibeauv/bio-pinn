# Classes
import sys
import os
sys.path.append(os.getcwd() + '\source')
from source.ode import ODE
from source.preprocess import Normalization
# PINN
import sciann as sn
# Other
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# Generate the artificial data
ode = ODE(1)
T = [1000]
A = 5
E = 1000
dt = 0.0001
tf = [0.0, 0.1, 0.2, 0.3, 0.4]
y0 = 1
# x1 : Temperature
# x2 : Reaction time
yA = ode.analytical_solution(T, A, E, tf, y0)

# Normalization
scaler = Normalization()
x, scaler_x = scaler.minmax(tf)

# PINN
x0 = 1
dt = 0.4
t = sn.Variable()
cA = sn.Functional("cA", [t], 4*[10], activation="tanh")
h = []
a = []
for i in range(2):
    h.append([])
    a.append([])
    for j in range(2):
        h[-1].append(sn.Functional('h'+str(i)+str(j), [t], 4*[10], activation="tanh"))
a = np.array([[0, 0], [0.5, 0.5]])
# Parameters
k = sn.Parameter(0.0, inputs=[t])
# ODE
L1 = (sn.math.add(h[0], a)*dt + x0)
L2 = (sn.math.add(a.dot(h[1]))*dt + x0)
# Supervised learning with data
DATA1 = cA
# Model
m = sn.SciModel([dt, t],
                [L1, L2, DATA1], optimizer="adagrad")
m.train([np.array(x)],
        ["zeros", "zeros", yA],
        epochs=5000,
        learning_rate=0.1)
# Predict
tf_pred = [0.12, 0.24, 0.28, 0.35, 0.5]
y_pred = cA.eval(m, [scaler_x.transform(np.array(tf_pred).reshape(-1,1))])
y_true = ode.analytical_solution(T, A, E, tf_pred, y0)
print(r2_score(y_pred, y_true))
# Visualization
plt.plot(tf_pred, y_true)
plt.plot(tf_pred, y_pred, 'o')
plt.legend(['Analytical Solution', 'Predictions'])
plt.xlabel('Time')
plt.ylabel('Concentration of A')
plt.show()