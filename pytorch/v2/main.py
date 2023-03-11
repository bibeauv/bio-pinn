from data import *
from pinn import *
import matplotlib.pyplot as plt
import torch.nn as nn

# Data
file = 'bio.csv'
device = torch.device('cpu')
t1 = 0
t2 = 8
mul = 100
colocation_points = (t2 - t1) * mul + 1

# Training set
X_train, Y_train, idx = read_data(file, device, t1, t2, colocation_points)

# Residual
f_hat = torch.zeros(X_train.shape[0], 1).to(device)

# PINN
neurons = 10
activation = nn.Tanh()
params = [1., 1., 1., 1., 1., 1.]
start_learning_rate = 1e-3

PINN_one_input = PINN(neurons, activation, params, idx, 
                      X_train, Y_train, f_hat, start_learning_rate, device)

# Training
epoch = 0
max_epochs = 100000
vec_loss = []
while epoch <= max_epochs:

    PINN_one_input.optimizer.step(PINN_one_input.closure)

    vec_loss.append(PINN_one_input.total_loss.detach().numpy())

    if epoch % 100 == 0:
        print(f'Epoch {epoch} \t loss_data: {PINN_one_input.loss_data:.4e} \t loss_ode: {PINN_one_input.loss_ode:.4e} \t loss_IC: {PINN_one_input.loss_IC:.4e}')

    if epoch == 5000:
        PINN_one_input.optimizer = torch.optim.Adam(PINN_one_input.params, lr=1e-4)

    if epoch == 50000:
        PINN_one_input.optimizer = torch.optim.Adam(PINN_one_input.params, lr=1e-5)

    epoch += 1

print(f'k1 = {PINN_one_input.nn.k1}')
print(f'k2 = {PINN_one_input.nn.k2}')
print(f'k3 = {PINN_one_input.nn.k3}')
print(f'k4 = {PINN_one_input.nn.k4}')
print(f'k5 = {PINN_one_input.nn.k5}')
print(f'k6 = {PINN_one_input.nn.k6}')

plt.plot(vec_loss)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

t = X_train.detach().numpy()

cB_pred = PINN_one_input.nn(X_train)[:,0].detach().numpy()
cTG_pred = PINN_one_input.nn(X_train)[:,1].detach().numpy()
cDG_pred = PINN_one_input.nn(X_train)[:,2].detach().numpy()
cMG_pred = PINN_one_input.nn(X_train)[:,3].detach().numpy()
cG_pred = PINN_one_input.nn(X_train)[:,4].detach().numpy()

cB_true = Y_train[0,0]
cTG_true = Y_train[idx,1]
cDG_true = Y_train[idx,2]
cMG_true = Y_train[idx,3]
cG_true = Y_train[0,4]

plt.plot(t, cB_pred, label='cB_pred')
plt.plot(t[0], cB_true, 'o', label='cB_true')
plt.xlabel('Time [min]')
plt.ylabel('Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t, cTG_pred, label='cTG_pred')
plt.plot(t[idx], cTG_true, 'o', label='cTG_true')
plt.xlabel('Time [min]')
plt.ylabel('Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t, cDG_pred, label='cDG_pred')
plt.plot(t[idx], cDG_true, 'o', label='cDG_true')
plt.xlabel('Time [min]')
plt.ylabel('Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t, cMG_pred, label='cMG_pred')
plt.plot(t[idx], cMG_true, 'o', label='cMG_true')
plt.xlabel('Time [min]')
plt.ylabel('Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t, cG_pred, label='cG_pred')
plt.plot(t[0], cG_true, 'o', label='cG_pred')
plt.xlabel('Time [min]')
plt.ylabel('Concentration [mol/L]')
plt.legend()
plt.show()
