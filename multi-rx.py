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

# Generate the artificial data
ode = ODE(2)
k = [1.5, 2]
dt = 0.0001
tf = [0, 0.1, 0.2, 0.3, 0.4]
y0 = 1
y, X = ode.euler_explicite_multi(k, dt, tf, y0)
#yA = get_features(y, 0)
#yA = np.array(yA).reshape(-1,1)
#yB = get_features(y, 1)
#yB = np.array(yB).reshape(-1,1)
yC = get_features(y, 2)
yC = np.array(yC).reshape(-1,1)

# Normalization
scaler = Normalization()
x, scaler_x = scaler.minmax(X)

# PINN
t = sn.Variable()
cA = sn.Functional("cA", [t], 4*[10], activation="tanh")
cB = sn.Functional("cB", [t], 4*[10], activation="tanh")
cC = sn.Functional("cC", [t], 4*[10], activation="tanh")
# Parameters
k1 = sn.Parameter(1.0, inputs=[t])
k2 = sn.Parameter(1.0, inputs=[t])
# ODE
L1 = sn.math.diff(cA, t) + k1*cA
L2 = sn.math.diff(cB, t) - k1*cA + k2*cB
L3 = sn.math.diff(cC, t) - k2*cB
# Supervised learning with data
DATA1 = (1-sn.utils.sign(t - 0.0001)) * (cA - 1)
DATA2 = (1-sn.utils.sign(t - 0.0001)) * (cB - 0)
DATA3 = cC
# Model
m = sn.SciModel([t],
                [L1, L2, L3, 
                 DATA1, 
                 DATA2, 
                 DATA3], optimizer="adagrad")
m.train([np.array(x)],
        ["zeros", "zeros", "zeros", 
         #np.array(yA), 
         "zeros", 
         #np.array(yB), 
         "zeros", 
         np.array(yC)],
        epochs=10000,
        learning_rate=0.1)
# Predict
tf_pred = [0.12, 0.24, 0.28, 0.35, 0.5]
yA_pred = cA.eval(m, [scaler_x.transform(np.array(tf_pred).reshape(-1,1))])
yB_pred = cB.eval(m, [scaler_x.transform(np.array(tf_pred).reshape(-1,1))])
yC_pred = cC.eval(m, [scaler_x.transform(np.array(tf_pred).reshape(-1,1))])
ode_pred = ODE(2)
y_true, _ = ode_pred.euler_explicite_multi(k, dt, tf_pred, y0)
yA_true = get_features(y_true, 0)
yB_true = get_features(y_true, 1)
yC_true = get_features(y_true, 2)
# Rate constants
k1_pred = k1.eval(m, [np.array(x)])
k2_pred = k2.eval(m, [np.array(x)])
print(k1_pred)
print(scaler_x.inverse_transform(np.array(k1_pred).reshape(-1,1)))
print(k2_pred)
print(scaler_x.inverse_transform(np.array(k2_pred).reshape(-1,1)))
# R2-score
print(r2_score(yA_pred, yA_true))
print(r2_score(yB_pred, yB_true))
print(r2_score(yC_pred, yC_true))
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