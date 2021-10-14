# Classes
import sys
import os

from tensorflow.python.keras import initializers
sys.path.append(os.getcwd() + '\source')
from source.ode import ODE
from source.preprocess import Normalization
# PINN
import sciann as sn
from tensorflow.keras.initializers import GlorotNormal
# Other
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Generate the artificial data
ode = ODE()
T = [1000, 1500, 2000]
A = 5
E = 1000
dt = 0.0001
tf = [0, 0.1, 0.2, 0.3, 0.4]
y0 = 1
# x1 : Temperature
# x2 : Reaction time
yA, yB, x1, x2 = ode.euler_explicite(T, A, E, dt, tf, y0)
for i in range(0, len(x1)):
    x1[i] = pow(x1[i], -1)

# Normalization
scaler = Normalization()
x1, scaler_x1 = scaler.minmax(x1)
x2, scaler_x2 = scaler.minmax(x2)
yA, scaler_yA = scaler.minmax(yA)
yB, scaler_yB = scaler.minmax(yB)

# PINN
T = sn.Variable()
t = sn.Variable()
cA = sn.Functional("cA", [T, t], 4*[10], activation="tanh", kernel_initializer=GlorotNormal)
# Parameters to predict
E_const = sn.Parameter(1.0, inputs=[T, t])
A_const = sn.Parameter(1.0, inputs=[T, t])
k = sn.utils.log(A_const) - E_const*T
# ODE
L1 = sn.math.diff(cA, t) + sn.utils.exp(k)*cA
# Initial condition
TOL = 0.001
IC = (1-sn.utils.sign(t - TOL)) * (cA - 1)
# Data
DATA = cA
# Model
m = sn.SciModel([T, t],
                [L1, IC, DATA], optimizer="adagrad")
# Train
m.train([np.array(x1), np.array(x2)],
        ["zeros", "zeros", np.array(yA)],
        epochs=10000,
        learning_rate=0.1)
# Predict
y_pred = cA.eval(m, [np.array(x1), np.array(x2)])
# Visualization
a = plt.axes(aspect='equal')
plt.scatter(scaler_yA.inverse_transform(yA), scaler_yA.inverse_transform(y_pred))
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 1]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()
print(r2_score(scaler_yA.inverse_transform(yA), scaler_yA.inverse_transform(y_pred)))