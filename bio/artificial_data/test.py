import sciann as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mul = 2

functionals = ['cA', 'cB', 'cC']

data = pd.read_csv('artificial_data.csv')
collocation_points = (len(data)-1)*mul+1

ids = []
y = []
i = 0
for col in data.columns:
    if col == 't':
        X = [np.linspace(data['t'].iloc[0],
                                data['t'].iloc[-1],
                                collocation_points).reshape(-1,1)]
    elif col == functionals[i]:
        y.append(data[functionals[i]].to_numpy())
        i += 1
    else:
        raise Exception("The functionals set for the PINN does not fit the data!")
        
for i in range(0,len(functionals)):
    ids += [(np.arange(0,collocation_points,mul).reshape(-1,1),
             y[i].reshape(-1,1))]

t = sn.Variable()

cA = sn.Functional("cA", [t], 4*[10], activation="tanh", kernel_initializer='glorot_uniform')
cB = sn.Functional("cB", [t], 4*[10], activation="tanh", kernel_initializer='glorot_uniform')
cC = sn.Functional("cC", [t], 4*[10], activation="tanh", kernel_initializer='glorot_uniform')

k1 = sn.Parameter(1, inputs=[t])
k2 = sn.Parameter(1, inputs=[t])

L1 = sn.math.diff(cA, t) + (k1)*cA
L2 = sn.math.diff(cB, t) - (k1)*cA + (k2)*cB
L3 = sn.math.diff(cC, t) - (k2)*cB

m = sn.SciModel([t],
                [L1, L2, L3,
                cA, cB, cC], optimizer='adagrad')

m.train(X,
        ['zeros', 'zeros', 'zeros'] + ids,
        epochs=1000,
        batch_size=collocation_points)

for i, evaluation in enumerate([cA, cB, cC]):
    y_pred = evaluation.eval(X)
    plt.plot(X[0], y_pred, label=evaluation.name+' pred')
    plt.plot(X[0].flatten()[ids[i][0].flatten()], y[i], 'o', label=evaluation.name+' true')
plt.legend()
plt.xlabel('Time [min]')
plt.ylabel('Concentration [mol/L]')
plt.show()