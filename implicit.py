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
y_t = []
y_tdt = []
X = []
for i in range(len(tf)):
        for j in range(i+1, len(tf)):
                X.append(tf[j]-tf[i])
                y_t.append(yA[i])
                y_tdt.append(yA[j])
yA_t = np.array(y_t).reshape(-1,1)
yA_tdt = np.array(y_tdt).reshape(-1,1)

# Normalization
scaler = Normalization()
x, scaler_x = scaler.minmax(X)

# PINN
dt = sn.Variable()
cA_t = sn.Functional("cA_t", [dt], 4*[10], activation="tanh")
cA_tdt = sn.Functional("cA_tdt", [dt], 4*[10], activation="tanh")
# Parameters
k = sn.Parameter(0.0, inputs=[dt])
# ODE
L1 = cA_tdt*(1+k*dt) - cA_t
# Supervised learning with data
DATA1 = cA_t
DATA2 = cA_tdt
# Model
m = sn.SciModel([dt],
                [L1, DATA1, DATA2], optimizer="adagrad")
m.train([np.array(x)],
        ["zeros", yA_t, yA_tdt],
        epochs=5000,
        learning_rate=0.1)
# Predict
tf_pred = [0.12, 0.24, 0.28, 0.35, 0.5]
y_pred = cA_t.eval(m, [scaler_x.transform(np.array(tf_pred).reshape(-1,1))])
y_true = ode.analytical_solution(T, A, E, tf_pred, y0)
print(r2_score(y_pred, y_true))
# Visualization
plt.plot(tf_pred, y_true)
plt.plot(tf_pred, y_pred, 'o')
plt.legend(['Analytical Solution', 'Predictions'])
plt.xlabel('Time')
plt.ylabel('Concentration of A')
plt.show()