from data import *
from pinn import *
from post_process import *
import matplotlib.pyplot as plt
import torch.nn as nn

# Data
GC_file = 'bio2.csv'
MW_file = 'lowT-mass-5W_2302071731344_MwData.csv'
device = torch.device('cpu')

# Training set
X_train, Y_train, idx = read_data(GC_file, MW_file, device)

# Residual
f_hat = torch.zeros(X_train.shape[0], 1).to(device)

# PINN
neurons = 10
activation = nn.Tanh()
params = []
for _ in range(6):
    k = np.ones((X_train.shape[0], 1))
    params.append(torch.from_numpy(k).float().to(device))
start_learning_rate = 1e-3

PINN_two_inputs = PINN(neurons, activation, params, idx, 
                       X_train, Y_train, f_hat, start_learning_rate, device)

# Training
for i, p in enumerate(PINN_two_inputs.nn.parameters()):
    p.data.clamp_(min=0.)

epoch = 0
max_epochs = 20000
vec_loss = []
while epoch <= max_epochs:

    PINN_two_inputs.optimizer.step(PINN_two_inputs.closure)

    PINN_two_inputs.nn.k1.data.clamp_(min=0.)
    # PINN_two_inputs.nn.k2.data.clamp_(min=0.)
    # PINN_two_inputs.nn.k3.data.clamp_(min=0.)
    # PINN_two_inputs.nn.k4.data.clamp_(min=0.)
    # PINN_two_inputs.nn.k5.data.clamp_(min=0.)
    # PINN_two_inputs.nn.k6.data.clamp_(min=0.)

    vec_loss.append(PINN_two_inputs.total_loss.detach().numpy())

    if epoch % 100 == 0:
        print(f'Epoch {epoch} \t loss_data: {PINN_two_inputs.loss_data:.4e} \t loss_ode: {PINN_two_inputs.loss_ode:.4e} \t loss_IC: {PINN_two_inputs.loss_IC:.4e}')

    if epoch == 5000:
        PINN_two_inputs.optimizer = torch.optim.Adam(PINN_two_inputs.params, lr=1e-4)

    #if epoch == X:
    #    PINN_two_inputs.optimizer = torch.optim.Adam(PINN_two_inputs.params, lr=1e-5)

    epoch += 1

# print(PINN_two_inputs.nn.k1)

plt.plot(vec_loss)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()

t = X_train[:,0].detach().numpy()

# cB_pred = PINN_two_inputs.nn(X_train)[:,0].detach().numpy()
cTG_pred = PINN_two_inputs.nn(X_train)[:,0].detach().numpy()
# cDG_pred = PINN_two_inputs.nn(X_train)[:,2].detach().numpy()
# cMG_pred = PINN_two_inputs.nn(X_train)[:,3].detach().numpy()
# cG_pred = PINN_two_inputs.nn(X_train)[:,4].detach().numpy()
# T_pred = PINN_two_inputs.nn(X_train)[:,1].detach().numpy()
k1_pred = PINN_two_inputs.nn.k1.detach().numpy()
# k2_pred = PINN_two_inputs.nn(X_train)[:,6].detach().numpy()
# k3_pred = PINN_two_inputs.nn(X_train)[:,7].detach().numpy()
# k4_pred = PINN_two_inputs.nn(X_train)[:,8].detach().numpy()
# k5_pred = PINN_two_inputs.nn(X_train)[:,9].detach().numpy()
# k6_pred = PINN_two_inputs.nn(X_train)[:,10].detach().numpy()

cB_true = Y_train[0,0]
cTG_true = Y_train[idx,1]
cDG_true = Y_train[idx,2]
cMG_true = Y_train[idx,3]
cG_true = Y_train[0,4]

y0 = np.array([0.570199408, 38.])
y_euler = euler_explicite(y0, t, k1_pred,
                          Y_train[:,6])

# plt.plot(t, cB_pred, label='cB_pred')
# plt.plot(t[0], cB_true, 'o', label='cB_true')
# plt.xlabel('Time [sec]')
# plt.ylabel('Concentration [mol/L]')
# plt.legend()
# plt.show()

plt.plot(t, cTG_pred, label='cTG_pred')
plt.plot(t[idx], cTG_true, 'o', label='cTG_true')
plt.plot(t, y_euler[:,0], '--', label='cTG_euler')
plt.xlabel('Time [sec]')
plt.ylabel('Concentration [mol/L]')
plt.legend()
plt.show()

plt.plot(t, y_euler[:,1], label='T_euler')
plt.plot(t, Y_train[:,5], '--', label='T_true')
# plt.plot(t, T_pred, label='T_pred')
plt.xlabel('Time [sec]')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# plt.plot(t, cDG_pred, label='cDG_pred')
# plt.plot(t[idx], cDG_true, 'o', label='cDG_true')
# plt.xlabel('Time [sec]')
# plt.ylabel('Concentration [mol/L]')
# plt.legend()
# plt.show()

# plt.plot(t, cMG_pred, label='cMG_pred')
# plt.plot(t[idx], cMG_true, 'o', label='cMG_true')
# plt.xlabel('Time [sec]')
# plt.ylabel('Concentration [mol/L]')
# plt.legend()
# plt.show()

# plt.plot(t, cG_pred, label='cG_pred')
# plt.plot(t[0], cG_true, 'o', label='cG_pred')
# plt.xlabel('Time [sec]')
# plt.ylabel('Concentration [mol/L]')
# plt.legend()
# plt.show()

plt.plot(Y_train[:,5].detach().numpy(), k1_pred, label='k1_pred')
# plt.plot(X_train[:,1].detach().numpy(), k2_pred, label='k2_pred')
# plt.plot(X_train[:,1].detach().numpy(), k3_pred, label='k3_pred')
# plt.plot(X_train[:,1].detach().numpy(), k4_pred, label='k4_pred')
# plt.plot(X_train[:,1].detach().numpy(), k5_pred, label='k5_pred')
# plt.plot(X_train[:,1].detach().numpy(), k6_pred, label='k6_pred')
plt.xlabel('Temperature')
plt.ylabel('Kinetic constant')
plt.show()
