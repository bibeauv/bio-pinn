# ODE
import sys
import os
sys.path.append(os.getcwd() + '\source')
from source.ode import ODE
# PINN
import sciann as sn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# Generate the artificial data
ode = ODE()
T = [900]
A = 5
E = 500
dt = 0.0001
tf = [0, 0.1, 0.2, 0.3, 0.4]
y0 = 1
# x1 : Temperature
# x2 : Reaction time
yB, x1, x2 = ode.euler_explicite(T, A, E, dt, tf, y0)
yA = []
for i in range(0, len(yB)):
    yA.append(y0 - yB[i])
for i in range(0, len(x1)):
    x1[i] = pow(x1[i], -1)

# Normalization
x1 = np.reshape(x1, (-1,1))
x2 = np.reshape(x2, (-1,1))
yA = np.reshape(yA, (-1,1))
yB = np.reshape(yB, (-1,1))
scaler_x1 = MinMaxScaler()
scaler_x2 = MinMaxScaler()
scaler_yB = MinMaxScaler()
scaler_yA = MinMaxScaler()
scaler_x1.fit(x1)
scaler_x2.fit(x2)
scaler_yB.fit(yB)
scaler_yA.fit(yA)
x1 = scaler_x1.transform(x1)
x2 = scaler_x2.transform(x2)
yB = scaler_yB.transform(yB)
yA = scaler_yA.transform(yA)

# PINN
Temperature = sn.Variable("T")
Reaction_Time = sn.Variable("tf")
cA = sn.Functional("cA", [Temperature, Reaction_Time], 8*[20], activation="tanh")
# Parameters to predict
E_const = sn.Parameter(0.0, inputs=[Temperature, Reaction_Time])
A_const = sn.Parameter(1.0, inputs=[Temperature, Reaction_Time])
k = A_const*sn.utils.exp(-E_const*Temperature)
# ODEs
L1 = sn.math.diff(cA, Reaction_Time) + k*cA
L2 = cA
# Model
m = sn.SciModel([Temperature, Reaction_Time],
                [L1, L2], optimizer="adagrad")
# Train
m.train([np.array(x1), np.array(x2)],
        ["zeros", np.array(yA)],
        epochs=5000,
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