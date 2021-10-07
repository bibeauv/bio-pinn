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
T = [750, 800, 850, 900]
A = 5
E = 500
dt = 0.0001
tf = [0, 0.1, 0.2, 0.3, 0.4]
y0 = 1
# x1 : Temperature
# x2 : Reaction time
yB, x1, x2 = ode.euler_explicite(T, A, E, dt, tf, y0)
    
# Normalization
x1 = np.reshape(x1, (-1,1))
x2 = np.reshape(x2, (-1,1))
yB = np.reshape(yB, (-1,1))
scaler_x1 = MinMaxScaler()
scaler_x2 = MinMaxScaler()
scaler_yB = MinMaxScaler()
scaler_x1.fit(x1)
scaler_x2.fit(x2)
scaler_yB.fit(yB)
x1 = scaler_x1.transform(x1)
x2 = scaler_x2.transform(x2)
yB = scaler_yB.transform(yB)

# PINN
Temperature = sn.Variable("T")
Reaction_Time = sn.Variable("tf")
cB = sn.Functional("cB", [Temperature, Reaction_Time], 8*[20], activation="tanh")
# ODEs
L4 = sn.Data(cB)
# Model
m = sn.SciModel([Temperature, Reaction_Time],
                [L4], optimizer="adam")
# Train
m.train([np.array(x1), np.array(x2)],
        [np.array(yB)],
        epochs=200,
        learning_rate=0.01)
# Predict
y_pred = cB.eval(m, [np.array(x1), np.array(x2)])
# Visualization
a = plt.axes(aspect='equal')
plt.scatter(scaler_yB.inverse_transform(yB), scaler_yB.inverse_transform(y_pred))
plt.xlabel('True Values')
plt.ylabel('Predictions')
lims = [0, 1]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()
print(r2_score(scaler_yB.inverse_transform(yB), scaler_yB.inverse_transform(y_pred)))