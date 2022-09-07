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

def get_features(y, pos):
    yi = [yi[pos] for yi in y]
    return yi

# Number of collocation points and temperatures
collocation_points = 101
vec_T = [1000, 1500, 1750, 2000]

# Time
tf = list(np.linspace(0,3,num=collocation_points))
tf = tf*len(vec_T)

# Temperatures
T = np.ones(len(tf))
j = 0
for i in range(0,len(T),collocation_points):
    T[i:i+collocation_points] = T[i:i+collocation_points]*vec_T[j]
    j += 1
T = list(T)

# Concentrations
ode = ODE(2)
y0 = 1
dt = 0.0001
yA = []
yB = []
yC = []
A = [2.5 ,3]
Ea = [500, 400]
for i in range(len(tf)):
    C = ode.euler_explicite_arrhenius(T[i], Ea, A, dt, tf[i], y0)
    yA.append(C[0])
    yB.append(C[1])
    yC.append(C[2])
yA = np.array(yA).reshape(-1,1)
yB = np.array(yB).reshape(-1,1)
yC = np.array(yC).reshape(-1,1)

# Normalization (or not)
scaler = Normalization()
#x, scaler_x = scaler.minmax(tf)
#X, scaler_X = scaler.minmax(T)
x = np.array(tf).reshape(-1,1)
X = np.power(np.array(T).reshape(-1,1), -1)

# PINN
t = sn.Variable()
Temp = sn.Variable()
cA = sn.Functional("cA", [t, Temp], 4*[10], activation="tanh")
cB = sn.Functional("cB", [t, Temp], 4*[10], activation="tanh")
cC = sn.Functional("cC", [t, Temp], 4*[10], activation="tanh")
# Parameters
Ea1 = sn.Parameter(510.0, inputs=[Temp])
Ea2 = sn.Parameter(410.0, inputs=[Temp])
A1 = sn.Parameter(1.0, inputs=[Temp])
A2 = sn.Parameter(1.0, inputs=[Temp])
k1 = sn.utils.log(A1) - Ea1*Temp
k2 = sn.utils.log(A2) - Ea2*Temp
# ODE
L1 = sn.math.diff(cA, t) + sn.utils.exp(k1)*cA
L2 = sn.math.diff(cB, t) - sn.utils.exp(k1)*cA + sn.utils.exp(k2)*cB
L3 = sn.math.diff(cC, t) - sn.utils.exp(k2)*cB
# Mass balance
L4 = cA + cB + cC - 1
# Model
m = sn.SciModel([t, Temp],
                [L1, L2, L3, L4,
                 cA, 
                 cB, 
                 cC], optimizer="adagrad")
m.train([x, X],
        ["zeros", "zeros", "zeros", "zeros",
         (np.arange(0,len(tf),10).reshape(-1,1), yA[0:len(tf):10]),
         (np.arange(0,len(tf),10).reshape(-1,1), yB[0:len(tf):10]), 
         (np.arange(0,len(tf),10).reshape(-1,1), yC[0:len(tf):10])],
        epochs=10000,
        #batch_size=400,
        learning_rate=0.1)
# Predict
tf_pred = list(np.linspace(0,3,num=10))
T_pred = np.ones(len(tf_pred))*1/1000
yA_pred = cA.eval(m, #[scaler_x.transform(np.array(tf_pred).reshape(-1,1)), scaler_X.transform(np.array(T_pred).reshape(-1,1))])
                 [np.array(tf_pred).reshape(-1,1), T_pred.reshape(-1,1)])
yB_pred = cB.eval(m, #[scaler_x.transform(np.array(tf_pred).reshape(-1,1)), scaler_X.transform(np.array(T_pred).reshape(-1,1))]) 
                 [np.array(tf_pred).reshape(-1,1), T_pred.reshape(-1,1)])
yC_pred = cC.eval(m, #[scaler_x.transform(np.array(tf_pred).reshape(-1,1)), scaler_X.transform(np.array(T_pred).reshape(-1,1))]) 
                 [np.array(tf_pred).reshape(-1,1), T_pred.reshape(-1,1)])
ode_pred = ODE(2)
yA_true = []
yB_true = []
yC_true = []
for i in range(len(tf_pred)):
    C = ode_pred.euler_explicite_arrhenius(1000, Ea, A, dt, tf_pred[i], y0)
    yA_true.append(C[0])
    yB_true.append(C[1])
    yC_true.append(C[2])
yA_true = np.array(yA_true).reshape(-1,1)
yB_true = np.array(yB_true).reshape(-1,1)
yC_true = np.array(yC_true).reshape(-1,1)
# Constants
print('Ea1 = ', Ea1.eval(m, [x, X]))
print('A1 = ', A1.eval(m, [x, X]))
print('Ea2 = ', Ea2.eval(m, [x, X]))
print('A2 = ', A2.eval(m, [x, X]))
# R2-score
print('r2_score yA = ', r2_score(yA_pred, yA_true))
print('r2_score yB = ', r2_score(yB_pred, yB_true))
print('r2_score yC = ', r2_score(yC_pred, yC_true))
# Visualization
plt.plot(tf_pred, yA_true)
plt.plot(tf_pred, yA_pred, 'o')
plt.plot(tf_pred, yB_true)
plt.plot(tf_pred, yB_pred, 'o')
plt.plot(tf_pred, yC_true)
plt.plot(tf_pred, yC_pred, 'o')
plt.legend(['Analytical Solution (A)', 'Predictions (A)',
            'Analytical Solution (B)', 'Predictions (B)',
            'Analytical Solution (C)', 'Predictions (C)'])
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.show()