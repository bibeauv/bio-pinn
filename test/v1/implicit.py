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
tf = list(np.linspace(0,3,num=5))
y0 = 1
# x1 : Temperature
# x2 : Reaction time
yA = ode.analytical_solution(T, A, E, tf, y0)
y_t = []
y_tdt = []
x_dt = []
x_t = []
for i in range(len(tf)):
        for j in range(i+1, len(tf)):
                x_dt.append(tf[j]-tf[i])
                x_t.append(tf[i])
                y_t.append(yA[i])
                y_tdt.append(yA[j])
yA_t = np.array(y_t).reshape(-1,1)
yA_tdt = np.array(y_tdt).reshape(-1,1)

# Normalization
scaler = Normalization()
x_dt, scaler_x_dt = scaler.minmax(x_dt)
x_t, scaler_x_t = scaler.minmax(x_t)

# PINN
dt = sn.Variable()
t = sn.Variable()
cA_t = sn.Functional("cA_t", [dt, t], 4*[10], activation="tanh")
cA_tdt = sn.Functional("cA_tdt", [dt, t], 4*[10], activation="tanh")
# Parameters
k = sn.Parameter(0.0, inputs=[dt, t])
# ODE
L1 = sn.math.diff(cA_tdt, dt) - (-k*cA_tdt)
# Supervised learning with data
DATA1 = cA_t
DATA2 = cA_tdt
# Model
m = sn.SciModel([dt, t],
                [L1, DATA1, DATA2], optimizer="adagrad")
m.train([np.array(x_dt), np.array(x_t)],
        ["zeros", yA_t, yA_tdt],
        epochs=10000,
        learning_rate=0.1)
# Predict
tf_pred = [0.05, 0.12, 0.24, 0.28, 0.35, 0.5, 0.65, 0.72, 0.89, 1.5, 1.6, 2.4, 3.6]
x_dt = []
x_t = []
for i in range(len(tf_pred)-1):
        x_dt.append(tf_pred[i+1]-tf_pred[i])
        x_t.append(tf_pred[i+1])
y_pred = cA_t.eval(m, [scaler_x_dt.transform(np.array(x_dt).reshape(-1,1)),
                       scaler_x_t.transform(np.array(x_t).reshape(-1,1))])
y_true = ode.analytical_solution(T, A, E, x_t, y0)
print(r2_score(y_pred, y_true))
# Visualization
plt.plot(x_t, y_true)
plt.plot(x_t, y_pred, 'o')
plt.legend(['Analytical Solution', 'Predictions'])
plt.xlabel('Time')
plt.ylabel('Concentration of A')
plt.show()