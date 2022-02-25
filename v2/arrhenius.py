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
ode = ODE(1)
T = [1000, 1500, 2000]
A = 5
E = 1000
dt = 0.0001
tf = np.linspace(0,3,100)
y0 = 1
# x1 : Temperature
# x2 : Reaction time
y, x = ode.euler_explicite(T, A, E, dt, tf, y0)
yA = [yA[0] for yA in y]
yB = [yB[1] for yB in y]
x1 = [x1[0] for x1 in x]
x2 = [x2[1] for x2 in x]
for i in range(0, len(x1)):
    x1[i] = pow(x1[i], -1)

# Normalization
scaler = Normalization()
#x1, scaler_x1 = scaler.minmax(x1)
x1 = np.array(x1).reshape(-1,1)
#x2, scaler_x2 = scaler.minmax(x2)
x2 = np.array(x2).reshape(-1,1)
#yA, scaler_yA = scaler.minmax(yA)
yA = np.array(yA).reshape(-1,1)
#yB, scaler_yB = scaler.minmax(yB)
yB = np.array(yB).reshape(-1,1)

# PINN
T = sn.Variable()
t = sn.Variable()
cA = sn.Functional("cA", [T, t], 4*[10], activation="tanh", kernel_initializer=GlorotNormal)
cB = sn.Functional("cB", [T, t], 4*[10], activation="tanh", kernel_initializer=GlorotNormal)
# Parameters to predict
E_const = sn.Parameter(1010.0, inputs=[T])
A_const = sn.Parameter(1.0, inputs=[T])
k = sn.utils.log(A_const) - E_const*T
# ODE
L1 = sn.math.diff(cA, t) + sn.utils.exp(k)*cA
L2 = sn.math.diff(cB, t) - sn.utils.exp(k)*cA
# Mass balance
#L3 = cA + cB - 1
# Model
m = sn.SciModel([T, t],
                [L1, L2, #L3,
                 cA,
                 cB], optimizer="adagrad")
# Train
m.train([x1, x2],
        ["zeros", "zeros", #"zeros",
        (np.arange(0,len(yA),10).reshape(-1,1), yA[0:len(yA):10]),
        (np.arange(0,len(yB),10).reshape(-1,1), yB[0:len(yB):10])],
        epochs=10000,
        learning_rate=0.1)
# Predict
yA_pred = cA.eval(m, [x1, x2])
yB_pred = cB.eval(m, [x1, x2])
print('Ea = ', E_const.eval(m, [x1, x2]))
print('A = ', A_const.eval(m, [x1, x2]))
#yA = scaler_yA.inverse_transform(yA)
#y_pred = scaler_yA.inverse_transform(y_pred)
# Visualization
a = plt.axes(aspect='equal')
plt.scatter(yA, yA_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 1]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()
print('r2 score yA = ', r2_score(yA, yA_pred))
print('r2 score yB = ', r2_score(yB, yB_pred))
# Concentration
fig, axs = plt.subplots(3,1)
# Concentration T1
axs[0].plot(tf, yA[:100])
axs[0].plot(tf, yA_pred[:100], '--')
axs[0].plot(tf, yB[:100])
axs[0].plot(tf, yB_pred[:100], '--')
axs[0].legend(['Real A', 'Pred A', 'Real B', 'Pred B'])
axs[0].set_xlabel('Temps')
axs[0].set_ylabel('Concentration')
# Concentration T2
axs[1].plot(tf, yA[100:200])
axs[1].plot(tf, yA_pred[100:200], '--')
axs[1].plot(tf, yB[100:200])
axs[1].plot(tf, yB_pred[100:200], '--')
axs[1].legend(['Real A', 'Pred A', 'Real B', 'Pred B'])
axs[1].set_xlabel('Temps')
axs[1].set_ylabel('Concentration')
# Concentration T3
axs[2].plot(tf, yA[200:])
axs[2].plot(tf, yA_pred[200:], '--')
axs[2].plot(tf, yB[200:])
axs[2].plot(tf, yB_pred[200:], '--')
axs[2].legend(['Real A', 'Pred A', 'Real B', 'Pred B'])
axs[2].set_xlabel('Temps')
axs[2].set_ylabel('Concentration')
plt.show()